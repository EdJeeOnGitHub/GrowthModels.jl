module NeuralGrowthModel

using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxCUDA
using Plots
using Sobol
using Statistics

export PositiveDense, 
       MicawberLayer, 
       SteadyStateLayer, 
       TechnologyLayer,
       generate_model_values, 
       generate_grid_values, 
       f_nn,
       f_nn_deriv,
       dfx, 
       predict_fn, 
       plot_pred_output, 
       plot_nn_output,
       loss,
       SolutionApproximation,
       ProjectionApproximation,
       UpwindApproximation,
       TrainHyperParams,
       train!,
       # 
       choose_device,
       check_gradients,
       draw_random_model,
       create_upwind_targets,
       check_statespace




abstract type SolutionApproximation end
struct ProjectionApproximation <: SolutionApproximation end
struct UpwindApproximation <: SolutionApproximation end




function extract_nn_parameters(::Type{M}, x::AbstractVecOrMat) where {M <: SkibaModel}
    γ, α, ρ, δ, A_H, A_L, κ = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :], x[7, :], x[8, :]
    return (γ = γ, α = α, ρ = ρ, δ = δ, A_H = A_H, A_L = A_L, κ = κ)
end

function extract_nn_parameters(::Type{M}, x::AbstractVecOrMat) where {M <: StochasticSkibaModel}
    return (
        γ = x[2, :], 
        α = x[3, :], 
        ρ = x[4, :], 
        δ = x[5, :], 
        A_H = x[6, :], 
        A_L = x[7, :], 
        κ = x[8, :], 
        θ = x[9, :], 
        σ = x[10, :]
    )
end

function extract_nn_parameters(::Type{M}, x::AbstractVecOrMat) where {M <: SmoothSkibaModel}
    γ, α, ρ, δ, A_H, A_L, κ, β = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :], x[7, :], x[8, :], x[9, :]
    return (γ = γ, α = α, ρ = ρ, δ = δ, A_H = A_H, A_L = A_L, κ = κ, β = β)
end

function extract_nn_parameters(::Type{M}, x::AbstractVecOrMat) where {M <: RamseyCassKoopmansModel}
    γ, α, ρ, δ, A = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :]
    return (γ = γ, α = α, ρ = ρ, δ = δ, A = A)
end


#### Defining Layers -----------------------------------------------------------
struct PositiveDense{F1, F2} <: Lux.AbstractExplicitLayer
    activation
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
end
function PositiveDense(in_dims::Int, out_dims::Int, activation = identity; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return PositiveDense{typeof(init_weight), typeof(init_bias)}(activation, in_dims, out_dims, init_weight, init_bias)
end

function Base.show(io::IO, d::PositiveDense)
    print(io, "PositiveDense($(d.in_dims) => $(d.out_dims))")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function Lux.initialparameters(rng::AbstractRNG, layer::PositiveDense)
    w = layer.init_weight(rng, layer.out_dims, layer.in_dims)
    b = layer.init_bias(rng, layer.out_dims, 1)
    return (weight = w, bias = b)
end

Lux.initialstates(::AbstractRNG, ::PositiveDense) = NamedTuple()
Lux.parameterlength(l::PositiveDense) = l.out_dims * l.in_dims + l.out_dims
Lux.statelength(::PositiveDense) = 0


@inline function (l::PositiveDense)(x::AbstractVecOrMat, ps, st::NamedTuple)
    # abs to ensure positive in input layer
    y = (abs.(ps.weight) * x) .+ ps.bias
    return l.activation.(y), st
end

abstract type GrowthModelLayer <: Lux.AbstractExplicitLayer end

struct MicawberLayer{F1, F2, T <: Model} <: GrowthModelLayer 
    activation
    state_size::Int
    m::T
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

struct SteadyStateLayer{F1, F2, T <: Model} <: GrowthModelLayer
    activation
    state_size::Int
    m::T
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end


struct TechnologyLayer{F1, F2, T <: Model} <: GrowthModelLayer
    activation
    state_size::Int
    m::T
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end


function TechnologyLayer(in_dims::Int, out_dims::Int, activation, m::Model, state_size = 1; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return TechnologyLayer{typeof(init_weight), typeof(init_bias), typeof(m)}(activation, state_size, m, in_dims, out_dims, init_weight, init_bias)
end

function MicawberLayer(in_dims::Int, out_dims::Int, activation, m::Model, state_size = 1; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return MicawberLayer{typeof(init_weight), typeof(init_bias), typeof(m)}(activation, state_size, m, in_dims, out_dims, init_weight, init_bias)
end


function SteadyStateLayer(in_dims::Int, out_dims::Int, activation, m::Model, state_size = 1; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return SteadyStateLayer{typeof(init_weight), typeof(init_bias), typeof(m)}(activation, state_size, m, in_dims, out_dims, init_weight, init_bias)
end

function Lux.initialparameters(rng::AbstractRNG, layer::GrowthModelLayer)
    w = layer.init_weight(rng, layer.out_dims, layer.state_size)
    b = layer.init_bias(rng, layer.out_dims, 1)
    return (weight = w, bias = b)
end


# if steady state layer or TechnologyLayer, then only one set of weights as 
# output of the layer is a scalar even if 2dim (i.e. k, z) inputs.
function Lux.initialparameters(rng::AbstractRNG, layer::Union{SteadyStateLayer, TechnologyLayer})
    w = layer.init_weight(rng, layer.out_dims, 1)
    b = layer.init_bias(rng, layer.out_dims, 1)
    return (weight = w, bias = b)
end

Lux.initialstates(::AbstractRNG, ::GrowthModelLayer) = NamedTuple()
Lux.parameterlength(l::GrowthModelLayer) = l.out_dims * l.in_dims + l.out_dims


function (l::MicawberLayer{F1, F2, M})(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple) where {F1, F2, T <: Real, M <: Model}
    states = x[1:l.state_size, :]
    ## WARNING: Currently just taking first input, since we know model params 
    # don't vary within batch
    params = extract_nn_parameters(M, x[:, 1])
    k_s = k_star(M; params...)
    y = (ps.weight * (states .- k_s)) .+ ps.bias
    return l.activation.(y), st
end


function (l::TechnologyLayer{F1, F2, M})(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple) where {F1, F2, T <: Real, M <: Model}
    states = x[1:l.state_size, :]
    ## WARNING: Currently just taking first input, since we know model params 
    # don't vary within batch
    params = extract_nn_parameters(M, x[:, 1])
    prod_output = production_function(M, states, params)
    y = (ps.weight * prod_output') .+ ps.bias
    return l.activation.(y), st
end



function (l::SteadyStateLayer{F1, F2, M})(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple) where {F1, F2, T <: Real, M <: Model}
    states = x[1:l.state_size, :]
    ## WARNING: Currently just taking first input, since we know model params 
    # don't vary within batch
    params = extract_nn_parameters(M, x[:, 1])
    k_ss = reduce(vcat, k_steady_state(M; params...))
    diff = -1 *(k_ss .- states[1, :]')
    abs_diff = abs.(diff)
    distances_indices = argmin(abs_diff, dims = 1)
    distances = diff[distances_indices]
    y = (ps.weight * distances) .+ ps.bias
    return l.activation.(y), st
end




#### Generating Candidate Points -----------------------------------------------
"""
generate_model_values(model_name)

Generate model hyperparameters using a SobolSequence.

# Arguments
- `model_name`: The type of the model.

# Returns
A SobolSequence object initialized with the lower bounds (`lb`) and upper bounds (`ub`) of the hyperparameters specific to the given `model_name`.
"""
function generate_model_values(model_type::Type{M}) where {M <: Model} 
    base_type = Base.typename(model_type).wrapper
    bounds_dict = Dict(
        SkibaModel => Dict(
            "ub" => [3.0, 0.9, 0.9, 0.5, 1.0, 1.0, 10.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        SmoothSkibaModel => Dict(
            "ub" => [10.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 1e5],
            "lb" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        RamseyCassKoopmansModel => Dict(
            "ub" => [10.0, 1.0, 1.0, 1.0, 20.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0]
        ),
        StochasticSkibaModel => Dict(
            "ub" => [3.0, 0.9, 0.9, 0.5, 1.0, 1.0, 10.0, 1.0, 3.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
    )

    lb = bounds_dict[base_type]["lb"]
    ub = bounds_dict[base_type]["ub"]
    sobol_seq = SobolSeq(lb, ub)

    return sobol_seq
end

"""
    generate_grid_values(m::StochasticModel, batch_size::Int; seed = 1234)

This function generates input values for the functions to be evaluated on.

# Arguments
- `m::StochasticModel`: The stochastic model.
- `batch_size::Int`: The number of input values to generate.
- `seed::Int`: (optional) The seed for the random number generator.

# Returns
- `vals::Array{Float32,2}`: The generated input values.

"""
function generate_grid_values(m::StochasticModel, batch_size::Int; seed = 1234)
    Random.seed!(seed)
    k_ss = k_steady_state(m)
    # k_st = min(GrowthModels.k_star(m), maximum(k_ss))
    k_st = maximum(k_ss);
    # model_centers = sample([k_ss; k_st], batch_size ÷ 2, replace = true)
    # model_points = abs.(randn(length(model_centers)) .* 0.01 .+ model_centers)
    sob_seq = SobolSeq(1e-5, 1.3*maximum(k_ss))
    grid_points = reduce(hcat, next!(sob_seq) for i = 1:(batch_size ÷ 2))
    # points = sort(vcat(model_points, grid_points[1, :]))
    points = sort(vcat(grid_points[1, :], grid_points[1, :]))
    m_vals = params(m)
    m_grid = repeat(m_vals, 1, batch_size)
    zs = exp.(randn(batch_size))
    vals = vcat(points', zs', m_grid)
    return Float32.(vals)
end


"""
    generate_grid_values(m::DeterministicModel, batch_size::Int; seed = 1234)

This function generates input values for the functions to be evaluated on.

# Arguments
- `m::DeterministModel`: The deterministic model.
- `batch_size::Int`: The number of input values to generate.
- `seed::Int`: (optional) The seed for the random number generator.

# Returns
- `vals::Array{Float32,2}`: The generated input values.

"""
function generate_grid_values(m::DeterministicModel, batch_size::Int; seed = 1234)
    Random.seed!(seed)
    k_ss = k_steady_state(m)
    # k_st = min(GrowthModels.k_star(m), maximum(k_ss))
    k_st = maximum(k_ss);
    # model_centers = sample([k_ss; k_st], batch_size ÷ 2, replace = true)
    # model_points = abs.(randn(length(model_centers)) .* 0.01 .+ model_centers)
    sob_seq = SobolSeq(1e-5, 1.3*maximum(k_ss))
    grid_points = reduce(hcat, next!(sob_seq) for i = 1:(batch_size ÷ 2))
    # points = sort(vcat(model_points, grid_points[1, :]))
    points = sort(vcat(grid_points[1, :], grid_points[1, :]))
    m_vals = params(m)
    m_grid = repeat(m_vals, 1, batch_size)
    vals = vcat(points', m_grid)
    return Float32.(vals)
end




"""
    f_nn(nn, k::AbstractVector, model_params, ps, st)

Utility function to evaluate the neural net.

Separates out the input states `k` and the macro model hyper parameters `model_params`.

# Arguments
- `nn`: The neural net model.
- `k`: Input states as an abstract vector.
- `model_params`: Macro model hyper parameters.
- `ps`: Additional parameters for the neural net evaluation.
- `st`: Additional state for the neural net evaluation.

# Returns
- The result of applying the neural net `nn` to the concatenated input states `k'` and `model_params`.

"""
function f_nn(nn, k::AbstractVector, model_params, ps, st)
    return first(Lux.apply(nn, vcat(k', model_params), ps, st))
end

"""
    f_nn(nn, k::AbstractMatrix, model_params, ps, st)

Utility function to evaluate the neural net.

Separates out the input states `k` and the macro model hyper parameters `model_params`.

# Arguments
- `nn`: The neural net model.
- `k`: Input states as an abstract matrix.
- `model_params`: Macro model hyper parameters.
- `ps`: Additional parameters for the neural net evaluation.
- `st`: Additional state for the neural net evaluation.

# Returns
- The result of applying the neural net `nn` to the concatenated input states `k` and `model_params`.

"""
function f_nn(nn, k::AbstractMatrix, model_params, ps, st)
    return first(Lux.apply(nn, vcat(k, model_params), ps, st))
end


# Calculate df/dx, returning f(x) and derivative of f(x) wrt x where x is a vector
# This isn't actually that useful as to use in NN need to calculate dfx/dparams and 
# zygote can't differentiate through itself. Therefore have to tell Zygote to forward 
# diff through this bit:
#
#     val, back = Zygote.pullback(k) do k_
#         Zygote.forwarddiff(k_) do k_
#            f(k_, model_params, ps, st)
#         end
#     end
# but this will take ~100x longer to actually calculate dfx
function dfx(f, k, model_params, ps, st)
    val, back = Zygote.pullback(k) do k_
            f(k_, model_params, ps, st)
    end
    return val, back(ones((1, size(k, 1))))[1]
end



"""
    f_nn_deriv(nn, v, k, model_params, ps, st; h = Float32(1e-3))

Compute the derivative of the neural network function `f_nn` with respect to `k` using finite difference method.
This function accounts for jumps in the function `f_nn` and reuses the value `v` at `k`.

# Arguments
- `nn`: Neural network object.
- `v`: Value of `f_nn` at `k`.
- `k`: Input value at which to compute the derivative.
- `model_params`: Model parameters.
- `ps`: Additional parameters.
- `st`: State variables.
- `h`: Step size for finite difference approximation. Default is `Float32(1e-3)`.

# Returns
- `diffs`: Array of computed derivatives.
"""
function f_nn_deriv(nn, v, k, model_params, ps, st; h = Float32(1e-3))
    fwd_v = f_nn(nn, k .+ h, model_params, ps, st)
    bwd_v = f_nn(nn, k .- h, model_params, ps, st)
    central_diff = (fwd_v .- bwd_v) ./ (2 * h)
    fwd_diff = (fwd_v .- v) ./ h
    bwd_diff = (v .- bwd_v) ./ h
    diffs = vcat(central_diff, bwd_diff, fwd_diff)
    abs_diffs = abs.(diffs)
    return diffs[argmin(abs_diffs, dims = 1)]
end
"""
    f_nn_deriv(nn, k, model_params, ps, st; h = Float32(1e-3))

Compute the derivative of the function `f_nn` with respect to `k` using finite differences.

# Arguments
- `nn`: Neural network
- `k`: Input value
- `model_params`: Model parameters
- `ps`: Additional parameters
- `st`: State variables
- `h`: Step size for finite differences (default: 1e-3)

# Returns
- `diffs`: Array of computed derivatives

"""
function f_nn_deriv(nn, k, model_params, ps, st; h = Float32(1e-3))
    v = f_nn(nn, k, model_params, ps, st)
    fwd_v = f_nn(nn, k .+ h, model_params, ps, st)
    bwd_v = f_nn(nn, k .- h, model_params, ps, st)
    central_diff = (fwd_v .- bwd_v) ./ (2 * h)
    fwd_diff = (fwd_v .- v) ./ h
    bwd_diff = (v .- bwd_v) ./ h
    diffs = vcat(central_diff, bwd_diff, fwd_diff)
    abs_diffs = abs.(diffs)
    return diffs[argmin(abs_diffs, dims = 1)]
end

function predict_fn(::UpwindApproximation, net::Chain, k, model_params, ps, st)
    return f_nn(net, k, model_params, ps, st)
end



function predict_fn(::ProjectionApproximation, net::Chain, k, model_params, ps, st; derivative = true)
    net_out = f_nn(net, k, model_params, nn_params[1], nn_states[1]) |> vec
    v_f_k = net_out[1, :]
    pol_f_k = net_out[2, :]
    if derivative
        v_f_deriv_k = f_nn_deriv(net, v_f_k, k, model_params, ps, st) |> vec
    else
        v_f_deriv_k = nothing
    end
    return v_f_k, v_f_deriv_k, pol_f_k
end


function check_statespace(m)
    hps = StateSpaceHyperParams(m)
    statespace = StateSpace(m, hps)
    max_statespace_constraint = statespace.aux_state.y[end] - m.δ * maximum(statespace[:k])
    min_statespace_constraint = statespace.aux_state.y[1] - m.δ * minimum(statespace[:k])
    state_error =  max_statespace_constraint < 0 || min_statespace_constraint < 0
    return state_error
end

function create_upwind_targets(sm::SolvedModel{T}, model_params, device) where {T <: DeterministicModel}
    upwind_pol = sm.variables[:c] |> device
    upwind_model_params = repeat(model_params[:, 1], 1, size(upwind_v, 1))
    upwind_states = sm.variables[:k] |> device
    return  (upwind_states,  upwind_pol), upwind_model_params
end

function create_upwind_targets(sm::SolvedModel{T}, model_params, device) where {T <: StochasticModel}
    upwind_pol = sm.variables[:c][:] |> device
    upwind_k = sm.variables[:k][:] |> device
    upwind_z = sm.variables[:z][:] |> device

    upwind_state = vcat(upwind_k', upwind_z')
    upwind_model_params = repeat(model_params[:, 1], 1, size(upwind_v, 1))

    return  (upwind_state,  upwind_pol), upwind_model_params
end
    
function loss(approx::UpwindApproximation, net, model_params, ps, st, targets)
    state_vals, upwind_pol = targets
    p_output = predict_fn(approx, net, state_vals, model_params, ps, st)
    n_k = length(p_output)
    upwind_pol_err = sqrt(sum(abs2, upwind_pol - vec(p_output)) / n_k)
    return upwind_pol_err
end

function error_projection(::Type{M}, state_vals, model_params, value_f, value_deriv_f, policy_f) where {M <: DeterministicModel}
    params = extract_nn_parameters(M, model_params)
    (; γ, ρ, δ) = params
    value_deriv_f = max.(value_deriv_f, Float32(1e-4))
    c = value_deriv_f .^ (-1 ./ γ)
    hjb_err = ρ .* value_f  .- (c .^ (1 .- γ)) ./ (1 .- γ) .- value_deriv_f .* (production_function(M, state_vals, params...) .- δ .* state_vals .- policy_f)
    policy_err = c .- policy_f
    return hjb_err, policy_err
end

function error_projection(::Type{M}, state_vals, model_params, value_f, value_deriv_f, policy_f) where {M <: StochasticModel}
    error("not yet implemented")
    # params = extract_nn_parameters(M, model_params)
    # (; γ, ρ, δ) = params
    # value_deriv_f = max.(value_deriv_f, Float32(1e-4))
    # c = value_deriv_f .^ (-1 ./ γ)
    # hjb_err = ρ .* value_f  .- (c .^ (1 .- γ)) ./ (1 .- γ) .- value_deriv_f .* (production_function(M, state_vals, params...) .- δ .* state_vals .- policy_f)
    # policy_err = c .- policy_f
    # return hjb_err, policy_err
end

function loss(approx::ProjectionApproximation, ::Type{M}, net, state_vals, model_params, ps, st) where {M <: Model}
    value_f, value_deriv_f, policy_f = predict_fn(approx::ProjectionApproximation, net, state_vals, model_params, ps, st)
    hjb_err, policy_err = error_projection(M, state_vals, model_params, value_f, value_deriv_f, policy_f)

    neg_deriv_penalty = sqrt(sum(exp.(min.(value_deriv_f, 0))) / length(value_deriv_f))
    l = sqrt(sum(abs2, policy_err) / length(policy_err)) + sqrt(sum(abs2, hjb_err) / length(hjb_err)) + neg_deriv_penalty
    return l
end

function draw_random_model(::ProjectionApproximation, ::Type{M}, sobol_seq) where {M <: Model}
    model_param_candidate = Float32.(param_reshuffle(next!(sobol_seq)))
    m = M(model_param_candidate...)
    max_ss = maximum(k_steady_state(m))
    state_constraint = check_statespace(m)
    redraw = !(max_ss < 25) || state_constraint 
    while redraw
        model_param_candidate = Float32.(param_reshuffle(next!(sobol_seq)))
        m = M(model_param_candidate...) 
        max_ss = maximum(k_steady_state(m))
        max_ss = maximum(k_steady_state(m))
        state_constraint = check_statespace(m)
        redraw = !(max_ss < 25) || state_constraint 
    end
    return m
end

function draw_random_model(::UpwindApproximation, ::Type{M}, sobol_seq) where {M <: DeterministicModel}
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



function draw_random_model(::UpwindApproximation, ::Type{M}, sobol_seq) where {M <: StochasticModel}
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


# Define a recursive function to check for 'NaN' values in nested named tuples
function check_gradients(obj)
    # Check if the object is a named tuple
    if typeof(obj) <: NamedTuple
        # Use any() to check if there are NaN values directly within current level of values
        if :weight in keys(obj) && any(isnan, obj[:weight])
            return true
        end
        # Recurse into deeper levels if no NaN is directly found
        for value in values(obj)
            if check_gradients(value)
                return true
            end
        end
    end
    return false  # Return false if no NaN values are found
end


# Define a function to fetch the appropriate device based on the hostname
function choose_device()
    # Get the current hostname
    host = gethostname()

    # Initialize the device variable
    device = nothing

    # Check if the hostname is "zero-gravitas"
    if host == "zero-gravitas"
        # Assign the CPU device if the condition is met
        device = cpu_device()
    else
        # Assign the GPU device otherwise
        device = gpu_device()
    end
    
    return device
end

param_reshuffle = function(p)
    new_p = copy(p)
    new_p[5] = p[5] + p[6]
    return new_p
end






function generate_values(m, batch_size, state_size, seed, device)
    random_vals = generate_grid_values(m, batch_size, seed = seed)
    state_vals, param_vals = random_vals[1:state_size, :], random_vals[state_size+1:end, :]
    state_vals = state_vals |> device
    param_vals = param_vals |> device
    return state_vals, param_vals
end


struct TrainHyperParams{T <: SolutionApproximation}
    approx::T
    n_redraw::Int
    batch_size::Int
    state_size::Int
    device::Union{LuxCUDADevice, LuxCPUDevice}
    m_type::Type
    sobol_seq::ScaledSobolSeq
    epoch_list::Vector{Int}
    loss_list::Vector
    print_iter::Int
    plot_iter::Int
    save_output::Bool
    save_path::String
    model_id::String
    display_fig::Bool
end

function TrainHyperParams(; approx = UpwindApproximation, n_redraw = 1, batch_size = 100, state_size = 1, device = choose_device(), m_type = StochasticSkibaModel, sobol_seq = generate_model_values(m_type), epoch_list = [1], loss_list = [Inf], print_iter = 100, plot_iter = 500, save_output = false,  save_path = "temp-data/",  model_id = "test-nn", display_fig = true)
    return TrainHyperParams(approx, n_redraw, batch_size, state_size, device, m_type, sobol_seq, epoch_list, loss_list, print_iter, plot_iter, save_output, save_path, model_id, display_fig)
end

function train!(curr_model::Model, epoch, st_opt, net::Chain, ps, last_ps, st, hps::TrainHyperParams{T}) where {T <: ProjectionApproximation}
    (; approx, n_redraw, batch_size, state_size, device, m_type, sobol_seq, epoch_list, 
        loss_list, print_iter, plot_iter, save_output, save_path, model_id, display_fig) = hps

    # redraw model every n_redraw epochs
    if epoch % n_redraw  == 0 || epoch == 1
        m = draw_random_model(approx, m_type, sobol_seq); 
    else
        m = curr_model
    end
    m = m |> device

    # generate state values to train on
    state_vals, param_vals = generate_values(m, batch_size, state_size, epoch, device)    
    # Calculate loss
    l, back = Zygote.pullback(ps) do p
        loss(approx, m_type, net, state_vals, param_vals, p, st)
    end;

    nan_loss_iter = 0
    while isnan(l)
        nan_loss_iter += 1
        println("Loss NaN, re-drawing model and using previous params.")
        m = draw_random_model(approx, m_type, sobol_seq); 
        state_vals, param_vals = generate_values(m, batch_size, state_size, epoch, device)    
        l, back = Zygote.pullback(last_ps) do p
            loss(approx, m_type, net, state_vals, param_vals, p, st)
        end;
        if nan_loss_iter > 50
            println("Too many NaN losses, skipping epoch.")
            break
        end
    end
    # if loss ok, update last_nn_params
    if !isnan(l)
        last_ps = deepcopy(ps)
    end


    if epoch % print_iter == 1
        println("Epoch: $epoch, Loss: $l")
    end
    push!(epoch_list, epoch)
    push!(loss_list, l)
  

    if epoch % plot_iter == 1
        try 
            if save_output
                output_dict = Dict(
                    :train_hps => train_hps,
                    :ps => ps,
                    :nn => nn,
                    :st => st)
                bson(
                    joinpath(
                        save_path,
                        "$model_id-epoch-$epoch-nn-output.bson"
                    ),  output_dict)

            end
            if display_fig
                # display(p_model_output)
            end
        catch e 
            println(e)
        end
    end

    safe_update!(l, back, loss_list, epoch, st_opt, ps)
end


function train!(curr_model::Tuple{Model, SolvedModel}, epoch, st_opt, net::Chain, ps, last_ps, st, hps::TrainHyperParams{T}) where {T <: UpwindApproximation}
    (; approx, n_redraw, batch_size, state_size, device, m_type, sobol_seq, epoch_list, 
        loss_list, print_iter, plot_iter, save_output, save_path, model_id, display_fig) = hps
    @show approx
    # redraw model every n_redraw epochs
    if epoch % n_redraw  == 0 || epoch == 1
        m, sm, _ = draw_random_model(approx, m_type, sobol_seq); 
    else
        m, sm = curr_model
    end
    m = m |> device

    # generate state values to train on
    _, param_vals = generate_values(m, batch_size, state_size, epoch, device)    
    # Create upwind "targets" from value function iteration
    upwind_targets, upwind_model_params = create_upwind_targets(sm, param_vals, device)
    # Calculate loss
    l, back = Zygote.pullback(ps) do p
        loss(approx, net, upwind_model_params, p, st, upwind_targets)
    end;

    nan_loss_iter = 0
    while isnan(l)
        nan_loss_iter += 1
        println("Loss NaN, re-drawing model and using previous params.")
        m, sm, _ = draw_random_model(approx, m_type, sobol_seq); 
        _, param_vals = generate_values(m, batch_size, state_size, epoch, device)    
        upwind_targets, upwind_model_params = create_upwind_targets(sm, param_vals, device)
        l, back = Zygote.pullback(last_ps) do p
            loss(approx, net, upwind_model_params, p, st, upwind_targets)
        end;
        if nan_loss_iter > 50
            println("Too many NaN losses, skipping epoch.")
            break
        end
    end
    # if loss ok, update last_nn_params
    if !isnan(l)
        last_ps = deepcopy(ps)
    end


    if epoch % print_iter == 1
        println("Epoch: $epoch, Loss: $l")
    end
    push!(epoch_list, epoch)
    push!(loss_list, l)
  

    if epoch % plot_iter == 1
        try 
            if save_output
                output_dict = Dict(
                    :train_hps => train_hps,
                    :ps => ps,
                    :nn => nn,
                    :st => st)
                bson(
                    joinpath(
                        save_path,
                        "$model_id-epoch-$epoch-nn-output.bson"
                    ),  output_dict)

            end
            if display_fig
                # display(p_model_output)
            end
        catch e 
            println(e)
        end
    end

    safe_update!(l, back, loss_list, epoch, st_opt, ps)
end


function safe_update!(loss, back, loss_list, epoch, st_opt, ps)
    if !isnan(loss) && loss < 1e10
        grads = back(1.0)[1]
        nan_grads = check_gradients(grads)
        if nan_grads 
            println("NaN Gradients")
        else
            # another attempt at avoiding huge loss swings leading to NaNs
            trailing_median_loss = NeuralGrowthModel.average_last_n(loss_list, min(epoch, 1000))
            loss_increase = loss / trailing_median_loss
            if loss_increase > 10_000
                @show loss_increase
                println("Loss Exploded, Skipping")
            else
                Optimisers.update!(st_opt, ps, grads)
            end
        end
    end
end



# function plot_pred_output(k_vals, v_f_k, v_f_deriv_k, pol_f_k)
#     p1 = plot(
#         vec(k_vals),
#         v_f_k,
#         colour = :blue,
#         label = "NN \$V(k)\$",
#         xlabel = "\$k\$",
#         ylabel = "\$V(k)\$",
#         )
#     p2 = plot(
#         vec(k_vals),
#         v_f_deriv_k,
#         colour = :blue,
#         label = "NN \$V'(k)\$",
#         xlabel = "\$k\$",
#         ylabel = "\$V'(k)\$",
#         )
#     p3 = plot(
#         vec(k_vals),
#         pol_f_k,
#         colour = :blue,
#         label = "NN \$c(k)\$",
#         xlabel = "\$k\$",
#         ylabel = "\$c(k)\$",
#         )
#     return p1, p2, p3
# end

# function moving_average(data, window_size)
#     # Calculate the moving average using a window of specified size
#     filter_length = length(data) - window_size + 1
#     ma = zeros(filter_length)
#     for i in 1:filter_length
#         ma[i] = sum(filter(!isfinite, filter(!isnan, data[i:i+window_size-1]))) / window_size
#     end
#     return ma
# end



# function average_last_n(vec::Vector, n::Int)
#     # Check if n is within the bounds of the vector length
#     if n > length(vec) || n < 1
#         return Inf
#     end

#     # Get the last n elements of the vector
#     last_n_elements = vec[end-n+1:end]

#     # Filter out Inf and NaN values
#     filtered_elements = filter(x -> !isinf(x) && !isnan(x), last_n_elements)

#     # If all elements are Inf or NaN, return Inf
#     if isempty(filtered_elements)
#         return Inf
#     end

#     # Calculate the average
#     return sum(filtered_elements)  / length(filtered_elements)
# end



# function plot_nn_output( 
#                         nets, 
#                         k_vals, 
#                         param_vals, 
#                         nn_params, 
#                         states, 
#                         epoch_list, 
#                         loss_list, 
#                         upwind_targets,
#                         cpu_dev,
#                         m::DeterministicModel
#                         ) 
#     upwind_k,  upwind_v, upwind_pol, upwind_kdot = upwind_targets

#     cpu_k_vals = k_vals |> cpu_dev
#     cpu_param_vals = param_vals |> cpu_dev
#     v_f_k, v_f_deriv_k, pol_f_k = predict_fn(nets, k_vals, param_vals, nn_params, states) |> cpu_dev
#     kdot = production_function(m, cpu_k_vals) .- cpu_param_vals[4, :] .* cpu_k_vals .- pol_f_k
#     hjb_err, pol_err = err_HJB(cpu_k_vals, cpu_param_vals, v_f_k, v_f_deriv_k, pol_f_k)

#     p1, p2, p3 = plot_pred_output(cpu_k_vals, v_f_k, v_f_deriv_k, pol_f_k)
#     plot!(p1, Array(upwind_k), Array(upwind_v), linewidth = 2, colour = :red, label = "Upwind")
#     plot!(p3, Array(upwind_k), Array(upwind_pol), linewidth = 2, colour = :red, label = "Upwind")

#     p4 = plot(
#         epoch_list, 
#         loss_list, 
#         label = "Loss", 
#         yscale = :log10,
#         alpha = 0.2
#         )

#     p5 = plot(
#         cpu_k_vals, 
#         kdot, 
#         label = "",
#         xlabel = "\$k\$",
#         ylabel = "\$\\dot{k}\$",
#         )
#     plot!(p5, Array(upwind_k), Array(upwind_kdot), label = "Upwind", colour = :red, linewidth = 2)
#     p6 = plot(
#         cpu_k_vals, 
#         hjb_err, 
#         seriestype = :scatter,
#         label = "",
#         xlabel = "\$k\$",
#         ylabel = "\$HJB Error\$",
#         )
#     p7 = plot(
#         cpu_k_vals, 
#         pol_err, 
#         seriestype = :scatter,
#         label = "",
#         xlabel = "\$k\$",
#         ylabel = "\$Policy Error\$",
#         )
#     return plot(p1, p2, p3, p4, p5, p6, p7, layout = (4, 2), size = (800, 800))
# end

# function plot_nn_output( 
#                         nets, 
#                         k_vals, 
#                         param_vals, 
#                         nn_params, 
#                         states, 
#                         epoch_list, 
#                         loss_list, 
#                         upwind_targets,
#                         cpu_dev,
#                         m::StochasticModel
#                         ) 

    

#     state_vals, upwind_v, upwind_pol, upwind_kdot = upwind_targets 
#     v_f_k, _, pol_f_k = predict_fn(nets, state_vals, param_vals, nn_params, states, derivative = false) 

#     v_kdot = production_function(m, state_vals[1, :], state_vals[2, :]) .- m.δ .* state_vals[1, :] .- pol_f_k 
#     v_kdot = Array(v_kdot) |> cpu_dev
#     u_states = Array(upwind_targets[1]) |> cpu_dev
#     v = Array(v_f_k) |> cpu_dev
#     pol = Array(pol_f_k) |> cpu_dev
#     nk = length(unique(u_states[1, :]))
#     nz = length(unique(u_states[2, :]))

#     group = repeat(collect(1:nz), inner = nk)

#     u_v = Array(upwind_targets[2])
#     u_pol = Array(upwind_targets[3])
#     u_kdot = Array(upwind_targets[4])

#     p1 = plot(
#         u_states[1, :], 
#         v_kdot,
#         group = group,
#         colour = :blue,
#         xlabel = "\$k\$",
#         ylabel = "\$\\dot(k)\$",
#         label = "",
#         title = "\$\\dot{k}\$"
#         )
#     plot!(
#         u_states[1, :],
#         u_kdot,
#         group = group,
#         colour = :red,
#         xlabel = "\$k\$",
#         ylabel = "\$\\dot(k)\$",
#         label = ""
#     )
    
#     p2 = plot(
#         u_states[1, :], 
#         v,
#         group = group,
#         colour = :blue,
#         xlabel = "\$k\$",
#         ylabel = "\$V(k)\$",
#         title = "Value Function"
#         )
#    plot!(
#     u_states[1, :],
#     u_v,
#         group = group,
#         colour = :red,
#         xlabel = "\$k\$",
#         ylabel = "\$V(k)\$",
#         label = ""
#    ) 

#    p3 = plot(
#     u_states[1, :],
#     pol,
#     group = group,
#     colour = :blue,
#     xlabel = "\$k\$",
#     ylabel = "\$c(k)\$",
#     label = "",
#     title = "Policy Function"
#    )
#    plot!(
#     u_states[1, :],
#     u_pol,
#     group = group,
#     colour = :red,
#     xlabel = "\$k\$",
#     ylabel = "\$c(k)\$",
#     label = ""
#    )

#    mean_loss = round(average_last_n(loss_list, length(epoch_list) ÷ 10), digits = 3)

#     p4 = plot(
#         epoch_list, 
#         loss_list, 
#         label = "Loss", 
#         yscale = :log10,
#         alpha = 0.2,
#         title = "Loss: $mean_loss"
#         )
#     roll_loss = rolling_mean(loss_list, min(2_000, length(loss_list)))
#     plot!(
#         p4, 
#         epoch_list, 
#         roll_loss, 
#         label = "Rolling Mean Loss", 
#         linewidth = 2,
#         alpha = 1.0
#         )
#     ylims!(p4, minimum(loss_list), 1e4)

#     return plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 800))
# end

# function rolling_mean(data::Vector, window_size::Int)
#     n = length(data)
#     result = fill(NaN, n)  # Initialize the result vector with NaNs

#     for i in 1:(n - window_size + 1)
#         # Extract the window of data
#         window = data[i:(i + window_size - 1)]
        
#         # Filter out NaN and infinite values
#         valid_values = filter(x -> !isnan(x) && isfinite(x) && x < 1e6, window)
        
#         # Compute the mean of the valid values
#         if !isempty(valid_values)
#             result[i] = median(valid_values) 
#         end
#     end

#     return result
# end

end