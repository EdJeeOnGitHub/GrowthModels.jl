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
cpu_dev = cpu_device()
device = gpu_device()



m_type = StochasticSkibaModel{Float32}
m = StochasticSkibaModel{Float32}() |> device
model_params = Float32.(params(m))
n_params = length(model_params)

batch_size = 200
using BenchmarkTools



function ValueFunctionChain(m::Model, c_size, n_params)
    state_size = ifelse(isa(m, StochasticModel), 2, 1)
    v_f_nn = Chain(
        # both read in simultaneously
        Parallel(
            nothing,
            MicawberLayer(state_size + n_params, c_size, relu, m),
            SteadyStateLayer(state_size + n_params, c_size, relu, m),
            NoOpLayer()
        ),
        x -> vcat(x...),
        Dense(c_size*2 + state_size + n_params, n_size, relu),
        Dense(n_size, n_size, relu),
        Dense(n_size, 1)
    )
    return v_f_nn
end

function PolicyFunctionChain(m::Model, c_size, n_params)
    state_size = ifelse(isa(m, StochasticModel), 2, 1)
    pol_f_nn = Chain(
        Parallel(
            nothing,
            MicawberLayer(state_size + n_params, c_size, relu, m),
            SteadyStateLayer(state_size + n_params, c_size, relu, m),
            NoOpLayer()
        ),
        x -> vcat(x...),
        Dense(c_size*2 + state_size + n_params, n_size, relu),
        Dense(n_size, n_size, relu),
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
param_reshuffle = function(p)
    new_p = copy(p)
    new_p[5] = p[5] + p[6]
    return new_p
end

function check_statespace(m)
    hps = StateSpaceHyperParams(m)
    statespace = StateSpace(m, hps)
    max_statespace_constraint = statespace.aux_state.y[end] - m.δ * maximum(statespace[:k])
    min_statespace_constraint = statespace.aux_state.y[1] - m.δ * minimum(statespace[:k])
    state_error =  max_statespace_constraint < 0 || min_statespace_constraint < 0
    return state_error
end

    



function projection_loss(nets, k, model_params, nn_params, states)
    v_f_k, v_f_deriv_k, pol_f_k = predict_fn(nets, k, model_params, nn_params, states)
    hjb_err, pol_err = err_HJB(k, model_params, v_f_k, v_f_deriv_k, pol_f_k)
    n_k = length(k)
    neg_deriv_penalty = sum(exp.(min.(v_f_deriv_k, 0)))
    loss = sqrt(sum(abs, pol_err) / n_k) + sqrt(sum(abs, hjb_err) / n_k) + sqrt(sum(abs, neg_deriv_penalty) / n_k)
    return loss
end

function GrowthModels.production_function(m::StochasticSkibaModel, state_vals::AbstractMatrix) 
    production_function(m, state_vals[1, :], state_vals[2, :])
end

function upwind_loss(nets, model_params, nn_params, states, upwind_targets, m::StochasticModel)
    state_vals, upwind_v, upwind_pol, upwind_kdot = upwind_targets
    v_f_k, _, pol_f_k = predict_fn(nets, state_vals, model_params, nn_params, states, derivative = false)
    kdot = production_function(m, state_vals[1, :], state_vals[2, :]) .- m.δ .* state_vals[1, :] .- pol_f_k

    n_k = length(v_f_k)
    upwind_v_err = sqrt(sum(abs2, upwind_v - v_f_k) / n_k)
    upwind_pol_err = sqrt(sum(abs2, upwind_pol - pol_f_k) / n_k)
    upwind_kdot_err = sqrt(sum(abs2, upwind_kdot - kdot) / n_k)
    return upwind_v_err + upwind_pol_err + upwind_kdot_err
end


function upwind_loss(nets, model_params, nn_params, states, upwind_targets, m::DeterministicModel)
    k, upwind_v, upwind_pol, upwind_kdot = upwind_targets
    v_f_k, _, pol_f_k = predict_fn(nets, k, model_params, nn_params, states, derivative = false)
    kdot = production_function(m, k) .- m.δ .* k .- pol_f_k

    n_k = length(v_f_k)
    upwind_v_err = sqrt(sum(abs2, upwind_v - v_f_k) / n_k)
    upwind_pol_err = sqrt(sum(abs2, upwind_pol - pol_f_k) / n_k)
    upwind_kdot_err = sqrt(sum(abs2, upwind_kdot - kdot) / n_k)
    return upwind_v_err + upwind_pol_err + upwind_kdot_err
end



function create_upwind_targets(sm::SolvedModel{T}, res, model_params) where {T <: DeterministicModel}
    upwind_v = res.value.v |> device
    upwind_pol = sm.variables[:c] |> device
    upwind_kdot = sm.kdot_function(sm.variables[:k]) |> device
    upwind_model_params = repeat(model_params[:, 1], 1, size(upwind_v, 1))
    upwind_k = sm.variables[:k] |> device
    return  (upwind_k, upwind_v, upwind_pol, upwind_kdot), upwind_model_params
end

function create_upwind_targets(sm::SolvedModel{T}, res, model_params) where {T <: StochasticModel}
    upwind_v = res.value.v[:] |> device
    upwind_pol = sm.variables[:c][:] |> device
    upwind_kdot = sm.kdot_function.(sm.variables[:k][:, 1], sm.variables[:z][1, :]')[:] |> device
    upwind_k = sm.variables[:k][:] |> device
    upwind_z = sm.variables[:z][:] |> device

    upwind_state = vcat(upwind_k', upwind_z')
    upwind_model_params = repeat(model_params[:, 1], 1, size(upwind_v, 1))

    return  (upwind_state, upwind_v, upwind_pol, upwind_kdot), upwind_model_params
end






function composite_loss(nets, k, model_params, nn_params, states, upwind_targets, m)
    proj_l = projection_loss(nets, k, model_params, nn_params, states)
    upwind_model_params = repeat(model_params[:, 1], 1, size(upwind_targets[2], 1))
    upwind_l = upwind_loss(nets, upwind_model_params, nn_params, states, upwind_targets, m)
    return proj_l + upwind_l
end

function draw_random_model(::Type{M}, sobol_seq) where {M <: DeterministicModel}
    m = M()
    max_ss = maximum(k_steady_state(m))
    state_constraint = check_statespace(m)
    successful_vfi = false
    redraw = !(max_ss < 25) || state_constraint || !successful_vfi
    while redraw
        model_param_candidate = Float32.(param_reshuffle(next!(sobol_seq)))


        m = M(model_param_candidate...) 
        max_ss = maximum(k_steady_state(m))
        try 
            sm, res = solve_growth_model(m, (Nk = 100,))
            successful_vfi = true
        catch e
            successful_vfi = false
            continue
        end
        max_ss = maximum(k_steady_state(m))
        state_constraint = check_statespace(m)
        redraw = !(max_ss < 25) || state_constraint || !successful_vfi
    end
    sm, res = solve_growth_model(m, (Nk = 100,))
    return m, sm, res
end



function draw_random_model(::Type{M}, sobol_seq) where {M <: StochasticModel}
    Nk = 50
    Nz = 2
    m = M()
    max_ss = maximum(k_steady_state(m))
    state_constraint = check_statespace(m)
    successful_vfi = false
    redraw = !(max_ss < 25) || state_constraint || !successful_vfi
    while redraw
        model_param_candidate = Float32.(param_reshuffle(next!(sobol_seq)))
        m = M(
            model_param_candidate[1:end-2]...,
            OrnsteinUhlenbeckProcess(θ = model_param_candidate[end-1], σ = model_param_candidate[end])
            )
        max_ss = maximum(k_steady_state(m))
        try 
            sm, res = solve_growth_model(m, (Nk = Nk, Nz = Nz))
            successful_vfi = true
        catch e
            successful_vfi = false
            continue
        end
        max_ss = maximum(k_steady_state(m))
        state_constraint = check_statespace(m)
        redraw = !(max_ss < 25) || state_constraint || !successful_vfi
    end
    sm, res = solve_growth_model(m, (Nk = Nk, Nz = Nz))
    return m, sm, res
end



epoch_list = [1]
loss_list = [Inf]
n_redraw = 1
# for epoch in epoch_list[end]:1_000_000
# epoch = 1
epoch = epoch_list[end]

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

    upwind_targets, upwind_model_params = create_upwind_targets(sm, res, param_vals)


    # state_vals, upwind_v, upwind_pol, upwind_kdot = upwind_targets
    # v_f_k, _, pol_f_k = predict_fn(nets, state_vals, upwind_model_params, nn_params, states, derivative = false)


    loss, back = Zygote.pullback(nn_params) do p
        # composite_loss(nets, k_vals, param_vals, p, states, upwind_targets, m)
        upwind_loss(nets, upwind_model_params, p, states, upwind_targets, m)
    end;

    if epoch % 10 == 1
        println("Epoch: $epoch, Loss: $loss")
    end
    push!(epoch_list, epoch)
    push!(loss_list, loss)
  


    if epoch % 50 == 1
        try 
            if (isa(m, StochasticModel))
                p_model_output = plot_nn_output(nets, k_vals, upwind_model_params, nn_params, states, epoch_list, loss_list, upwind_targets, cpu_dev, m)
            else
                p_model_output = plot_nn_output(nets, k_vals, param_vals, nn_params, states, epoch_list, loss_list, upwind_targets, cpu_dev, m)
            end
            display(p_model_output)
        catch e 
            println(e)
        end
    end
    if !isnan(loss) && loss < 1e10
        grads = back(1.0)[1]
        if any(isnan, grads[1][1][1].weight) || any(isnan, grads[2][1][1].weight)
            println("NaN Gradients")
        else
            Optimisers.update!(st_opt, nn_params, grads)
        end
    end;
end;


length(epoch_list)
savefig("skiba-nn-fit.pdf")





v_f_k, v_f_deriv_k, pol_f_k = predict_fn(fns, nets, cu(cpu_k_vals), cu(cpu_param_vals), nn_params[1], nn_params[2], states[1], states[2])



kdot = production_function(m, k_vals) .- m.δ .* k_vals .- pol_f_k


skiba_hyperparams = StateSpaceHyperParams(SkibaModel())
skiba_state = StateSpace(SkibaModel(), skiba_hyperparams)
skiba_init_value = Value(skiba_state);

fit_value, fit_variables, fit_iter = solve_HJB(
    SkibaModel(), 
    skiba_hyperparams, 
    init_value = skiba_init_value, maxit = 1000);
m = SkibaModel()
sm = SolvedModel(SkibaModel(), fit_value, fit_variables)

using DataInterpolations

sm_v_interp = DataInterpolations.LinearInterpolation(fit_value.v, fit_variables.k)
sm_v_deriv_interp = DataInterpolations.LinearInterpolation(fit_value.dVf, fit_variables.k)

sm_v_f_k = sm_v_interp(vec(cpu_k_vals))
sm_v_f_deriv_k = sm_v_deriv_interp(vec(cpu_k_vals))
sm_pol_f_k = sm.policy_function(vec(cpu_k_vals))

sm_h_err, sm_p_err = err_HJB(cpu_k_vals, cpu_param_vals , sm_v_f_k, sm_v_f_deriv_k, sm_pol_f_k)

nn_h_err, nn_p_err = err_HJB(vec(k_vals), param_vals, v_f_k, v_f_deriv_k, pol_f_k)

sum(abs2, sm_h_err)
sum(abs2, sm_p_err)

sum(abs2, nn_h_err)
sum(abs2, nn_p_err)



p1, p2, p3 = plot_pred_output(cpu_k_vals, Array(v_f_k), Array(v_f_deriv_k), Array(pol_f_k))
## Adding upwind solution for comparison
plot!(
    p1, 
    cpu_k_vals, 
    sm_v_f_k, 
    seriestype = :scatter, 
    label = "Upwind \$V(k)\$", 
    colour = :red)
plot!(
    p2,
    cpu_k_vals,
    sm_v_f_deriv_k,
    seriestype = :scatter,
    label = "Upwind \$V'(k)\$",
    colour = :red
)
plot!(
    p3,
    cpu_k_vals,
    sm_pol_f_k,
    seriestype = :scatter,
    label = "Upwind \$c(k)\$",
    colour = :red)

p4 = plot(
    epoch_list, 
    loss_list, 
    label = "Loss", 
    yscale = :log10,
    ylabel = "MSE",
    xlabel = "Epochs"
    )
p5 = plot(
    cpu_k_vals, 
    Array(kdot), 
    seriestype = :scatter,
    label = "NN \$\\dot{k}\$",
    xlabel = "\$k\$",
    ylabel = "\$\\dot{k}\$",
    )
plot!(
    p5,
    cpu_k_vals,
    sm.kdot_function(cpu_k_vals),
    seriestype = :scatter,
    label = "Upwind \$\\dot{k}\$",
    xlabel = "\$k\$",
    ylabel = "\$\\dot{k}\$"
)

p6 = plot(
    cpu_k_vals, 
    Array(hjb_err), 
    seriestype = :scatter,
    label = "",
    xlabel = "\$k\$",
    ylabel = "\$HJB Error\$",
    )
p7 = plot(
    cpu_k_vals, 
    Array(pol_err), 
    seriestype = :scatter,
    label = "",
    xlabel = "\$k\$",
    ylabel = "\$Policy Error\$",
    )
p_all = plot(p1, p2, p3, p4, p5, p6, p7, layout = (4, 2), size = (800, 800))

savefig(p_all, "skiba-nn-fit.pdf")

using Plots
plot(
    vec_vals, v_f_k, 
    xlabel = "K",
    ylabel = "Vf(K)",
    title = "Value Function",
    seriestype = :scatter,
    label = ""
)
plot!(
    vec_vals,
    sm_v_f_k,
    seriestype = :scatter,
    label ="",
    colour = :red
)
plot(
    vec_vals, v_f_deriv_k, 
    xlabel = "K",
    ylabel = "Vf(K)",
    title = "Value Function",
    seriestype = :scatter,
    label = ""
)
plot(
    vec_vals, pol_f_k,
    xlabel = "K",
    ylabel = "c(K)",
    title = "Policy Function",
    seriestype = :scatter,
    label = ""
)
plot!(
    vec_vals, sm_pol_f_k,
    #  label = "Policy Function",
    xlabel = "K",
    ylabel = "c(K)",
    title = "Policy Function",
    seriestype = :scatter,
    label = ""
)
plot!(
    vec_vals, sm_pol_f_k,
    xlabel = "K",
    ylabel = "c(K)",
    title = "Policy Function",
    seriestype = :scatter,
    label = ""
)
using BenchmarkTools
if benchmark
    @btime v_f(v_f_nn, k_vals, param_vals, vf_ps, vf_st);
    @btime v_f(v_f_nn, vals, vf_ps, vf_st);
    @btime v_f_deriv(v_f_nn, k_vals, param_vals, vf_ps, vf_st);
    @btime dfx(v_f, k_vals, param_vals, vf_ps, vf_st);
end




if benchmark
    @btime l, b = Zygote.pullback(vf_ps) do p
        v_f_deriv(v_f_nn, k_vals, param_vals, p, vf_st)
    end;
    @btime l, b = Zygote.pullback(vf_ps) do p
        dfx(v_f, k_vals, param_vals, p, vf_st)
    end;
end

if benchmark
    @btime l, back = Zygote.pullback(nn_params) do p
        loss_fn(fns, nets, k_vals, param_vals, p[1], p[2], states[1], states[2])
    end;
end