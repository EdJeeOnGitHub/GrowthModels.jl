using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxCUDA
using Plots
using Sobol
using BSON


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

function GrowthModels.production_function(::Type{M}, states::AbstractArray, params) where {M <: SkibaModel}
    production_function(M; k = states, params...)'
end

cpu_dev = cpu_device()
device = choose_device()



m_type = SkibaModel{Float32}
m = SkibaModel{Float32}() |> device
state_size = ifelse(isa(m, StochasticModel), 2, 1)
model_params = Float32.(params(m))
n_params = length(model_params)

batch_size = 100

function ValuePolicyFunctionChain(m::Model, c_size, n_params)
    state_size = ifelse(isa(m, StochasticModel), 2, 1)
    nn = Chain(
        # both read in simultaneously
        Parallel(
            nothing,
            MicawberLayer(state_size + n_params, c_size, relu, m, state_size),
            SteadyStateLayer(state_size + n_params, c_size, relu, m, state_size),
            TechnologyLayer(state_size + n_params, c_size, relu, m, state_size),
            NoOpLayer()
        ),
        x -> vcat(x...),
        Dense(c_size*3 + state_size + n_params, n_size, relu),
        Dense(n_size, n_size, relu),
        Dense(n_size, 2)
    )
    return nn
end

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
        Dense(c_size*3 + state_size + n_params, n_size, relu),
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
            MicawberLayer(state_size + n_params, c_size, tanh, m, state_size),
            SteadyStateLayer(state_size + n_params, c_size, tanh, m, state_size),
            TechnologyLayer(state_size + n_params, c_size, relu, m, state_size),
            NoOpLayer()
        ),
        x -> vcat(x...),
        Dense(c_size*3 + state_size + n_params, n_size, tanh),
        Dense(n_size, n_size, tanh),
        Dense(n_size, 1, softplus)
    )
    return pol_f_nn
end


c_size = 24
n_size = 48
nn = ValuePolicyFunctionChain(m, c_size, n_params)

# Creating input variables for testing
function setup_nn_hps(nn, rng, device)
    ps, st = Lux.setup(rng, nn) .|> device
    opt = Optimisers.ADAM()
    st_opt = Optimisers.setup(opt, ps) |> device
    return nn, ps, st, st_opt
end





approx = ProjectionApproximation()

sobol_seq = generate_model_values(m_type)
train_hps = TrainHyperParams(
    m_type,
    state_size,
    device;
    approx = approx
    )

rng = Random.default_rng();
nn, ps, st, st_opt = setup_nn_hps(nn, rng, device);
last_ps = deepcopy(ps);

m  = draw_random_model(approx, m_type, sobol_seq);
curr_m = m |> device;
# warmup check
train!(curr_m, 1, st_opt, nn, ps, last_ps, st, train_hps);


for epoch in train_hps.epoch_list[end]:25_000_000
    train!(curr_m, epoch, st_opt, nn, ps, last_ps, st, train_hps)
end;

input_values = generate_grid_values(m, 100)
k_values = input_values[1, :] |> device
param_values = input_values[2:end, :] |> device

value_f, value_deriv_f, policy_f = predict_fn(approx, nn, k_values, param_values, ps, st) |> cpu_dev

plot(
    Array(k_values),
    value_f
)


sm, res = solve_growth_model(m)

plot(
    sm.variables[:k],
    res.value.v[:]
)
plot!(
    Array(k_values),
    value_f
)

output_dict = Dict(
    :train_hps => train_hps,
    :ps => ps,
    :nn => nn,
    :st => st)
bson("nn_output.bson",  output_dict)

savefig("skiba-nn-fit.pdf")
