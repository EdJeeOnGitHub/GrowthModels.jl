using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxCUDA
using Plots
using Sobol

include("SolveNeuralModel.jl")
using .NeuralGrowthModel
benchmark = false
CUDA.allowscalar(false)

Random.seed!(1234)


# Dispatching for NN specific stuff
function GrowthModels.production_function(::Type{M}, states::AbstractMatrix, params) where {M <: StochasticSkibaModel}
    production_function(M; k = states[1, :], z = states[2, :], params...)
end
function GrowthModels.production_function(m::StochasticSkibaModel, state_vals::AbstractMatrix) 
    production_function(m, state_vals[1, :], state_vals[2, :])
end

cpu_dev = cpu_device()
device = choose_device()



m_type = StochasticSkibaModel{Float32}
m = StochasticSkibaModel{Float32}() |> device
model_params = Float32.(params(m))
n_params = length(model_params)

batch_size = 100


function ValueFunctionChain(m::Model, c_size, n_params)
    state_size = ifelse(isa(m, StochasticModel), 2, 1)
    v_f_nn = Chain(
        # both read in simultaneously
        Parallel(
            nothing,
            MicawberLayer(state_size + n_params, c_size, relu, m, state_size),
            SteadyStateLayer(state_size + n_params, c_size, relu, m, state_size),
            TechnologyLayer(state_size + n_params, c_size, relu, m, state_size),
            NoOpLayer()
        ),
        x -> vcat(x...),
        # BatchNorm(c_size*3 + state_size + n_params, relu, track_stats = false),
        Dense(c_size*3 + state_size + n_params, n_size, relu),
        # BatchNorm(n_size, relu, track_stats = false),
        Dense(n_size, n_size, relu),
        # BatchNorm(n_size, relu, track_stats = false),
        Dense(n_size, 1)
    )
    return v_f_nn
end

function PolicyFunctionChain(m::Model, c_size, n_params)
    state_size = ifelse(isa(m, StochasticModel), 2, 1)
    pol_f_nn = Chain(
        Parallel(
            nothing,
            MicawberLayer(state_size + n_params, c_size, tanh, m, state_size),
            SteadyStateLayer(state_size + n_params, c_size, tanh, m, state_size),
            TechnologyLayer(state_size + n_params, c_size, relu, m, state_size),
            NoOpLayer()
        ),
        x -> vcat(x...),
        # BatchNorm(c_size*3 + state_size + n_params, relu, track_stats = false),
        Dense(c_size*3 + state_size + n_params, n_size, tanh),
        # BatchNorm(n_size, relu, track_stats = false),
        Dense(n_size, n_size, tanh),
        # BatchNorm(n_size, relu, track_stats = false),
        Dense(n_size, 1, softplus)
    )
    return pol_f_nn
end


state_size = ifelse(isa(m, StochasticModel), 2, 1)
c_size = 24
n_size = 48

v_f_nn = ValueFunctionChain(m, c_size, n_params)
pol_f_nn = PolicyFunctionChain(m, c_size, n_params)
state_size = ifelse(isa(m, StochasticModel), 2, 1)

# Creating input variables for testing
vals = generate_grid_values(m, batch_size)
cpu_vals = deepcopy(vals)
cpu_k_vals = vec(cpu_vals[1:state_size, :])
cpu_param_vals = cpu_vals[state_size + 1:end, :]
vals = vals |> device
state_vals = vals[1:state_size, :]
state_vals = ifelse(isa(state_vals, Vector), vec(state_vals), state_vals)
k_vals = state_vals[1, :]
param_vals = vals[state_size + 1:end, :]
# Setting up NN params
rng = Random.default_rng()
# Version on the CPU
cpu_vf_ps, cpu_vf_st = Lux.setup(rng, v_f_nn) 
cpu_pol_ps, cpu_pol_st = Lux.setup(rng, pol_f_nn) 

vf_ps, vf_st = Lux.setup(rng, v_f_nn) .|> device
pol_ps, pol_st = Lux.setup(rng, pol_f_nn) .|> device

states = (vf_st, pol_st) |> device
nn_params = (vf_ps, pol_ps) |> device
opt = Optimisers.ADAM() 
st_opt = Optimisers.setup(ADAM(), nn_params) |> device

nets = (v_f_nn, pol_f_nn)

v_f_k, v_f_deriv_k, pol_f_k = predict_fn(nets, state_vals, param_vals, nn_params, states)




skiba_sobol_seq = generate_model_values("StochasticSkibaModel")

# make sure A_H > A_L


epoch_list = [1]
loss_list = [Inf]
n_redraw = 1
for epoch in epoch_list[end]:1_000_000
# epoch = 1
    if epoch == 1 
        last_nn_params = deepcopy(nn_params)
    end 
    if epoch % n_redraw  == 0 || epoch == 1
        m, sm, res = draw_random_model(m_type, skiba_sobol_seq); 
    end
    m = m |> device
    random_vals = generate_grid_values(m, batch_size, seed = epoch) 

    k_vals, param_vals = random_vals[1:state_size, :], random_vals[state_size+1:end, :]

    cpu_k_vals = deepcopy(k_vals)
    cpu_param_vals = deepcopy(param_vals)

    k_vals = k_vals |> device
    param_vals = param_vals |> device

    upwind_targets, upwind_model_params = create_upwind_targets(sm, res, param_vals, device)


    # state_vals, upwind_v, upwind_pol, upwind_kdot = upwind_targets
    # v_f_k, _, pol_f_k = predict_fn(nets, state_vals, upwind_model_params, nn_params, states, derivative = false)


    loss, back = Zygote.pullback(nn_params) do p
        # composite_loss(nets, k_vals, param_vals, p, states, upwind_targets, m)
        upwind_loss(nets, upwind_model_params, p, states, upwind_targets, m)
    end;

    # if loss becomes NaN, go back to last set of params, otherwise save them 
    # for next iteration
    if isnan(loss)
        println("Loss NaN, using previous params and re-running.")
        loss, back = Zygote.pullback(last_nn_params) do p
            upwind_loss(nets, upwind_model_params, p, states, upwind_targets, m)
        end;
    end

    if !isnan(loss)
        last_nn_params = deepcopy(nn_params)
    end


    if epoch % 10 == 1
        println("Epoch: $epoch, Loss: $loss")
    end
    push!(epoch_list, epoch)
    push!(loss_list, loss)
  


    if epoch % 500 == 1
        try 
            if (isa(m, StochasticModel))
                p_model_output = plot_nn_output(nets, k_vals, upwind_model_params, nn_params, states, epoch_list, loss_list, upwind_targets, cpu_dev, m)
            else
                p_model_output = plot_nn_output(nets, k_vals, param_vals, nn_params, states, epoch_list, loss_list, upwind_targets, cpu_dev, m)
            end
            savefig(p_model_output, "temp-data/nn-fit.pdf")
        catch e 
            println(e)
        end
    end

    # Ignore very large losses to prevent propagating NaNs 
    if !isnan(loss)  && loss < 1e10
        grads = back(1.0)[1]
        nan_v_grads = check_gradients(grads[1])
        nan_pol_grads = check_gradients(grads[2])
        if nan_v_grads || nan_pol_grads
            println("NaN Gradients")
        else
            # another attempt at avoiding huge loss swings leading to NaNs
            trailing_median_loss = NeuralGrowthModel.average_last_n(loss_list, min(epoch, 1000))
            loss_increase = loss / trailing_median_loss
            if loss_increase > 10_000
                @show loss_increase
                println("Loss Exploded, Skipping")
            else
                Optimisers.update!(st_opt, nn_params, grads)
            end
        end
    end;
end;


using BSON
output_dict = Dict(
    :epoch_list => epoch_list,
    :loss_list => loss_list,
    :nn_params => nn_params,
    :v_f_nn => v_f_nn,
    :pol_f_nn => pol_f_nn,
    :states => states)
bson("nn_output.bson",  output_dict)

savefig("skiba-nn-fit.pdf")
