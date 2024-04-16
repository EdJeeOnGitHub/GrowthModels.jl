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
       err_HJB, 
       monotonicity_penalty, 
       calculate_lipschitz_constant, 
       generate_model_values, 
       generate_grid_values, 
       f_nn,
       f_nn_deriv,
       dfx, 
       predict_fn, 
       plot_pred_output, 
       plot_nn_output,
       # 
       choose_device,
       check_gradients,
       draw_random_model,
       create_upwind_targets,
       composite_loss,
       upwind_loss,
       projection_loss,
       check_statespace








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
    param_vec = reduce(vcat, [params...])
    prod_output = production_function(M, states,  param_vec)
    y = (ps.weight * prod_output') .+ ps.bias
    return l.activation.(y), st
end


# k_steady_state_hi_Skiba(α::Real, A_H::Real, ρ::Real, δ::Real, κ::Real) = (α*A_H/(ρ + δ))^(1/(1-α)) + κ
# k_steady_state_lo_Skiba(α::Real, A_L::Real, ρ::Real, δ::Real) = (α*A_L/(ρ + δ))^(1/(1-α))

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


function err_HJB(::Type{M}, k, model_params, v_f_k, v_f_deriv_k, pol_f_k) where {M <: DeterministicModel}
    params = extract_nn_parameters(M, model_params)
    (; γ, α, ρ, δ, A_H, A_L, κ) = params
    v_f_deriv_k = max.(v_f_deriv_k, Float32(1e-4))
    c = v_f_deriv_k .^ (-1 ./ γ)
    hjb_err = ρ .* v_f_k  .- (c .^ (1 .- γ)) ./ (1 .- γ) .- v_f_deriv_k .* (production_function(M, k, params...) .- δ .* k .- pol_f_k)
    pol_err = c .- pol_f_k
    return hjb_err, pol_err
end

function err_HJB(::Type{M}, k, z, model_params, v_f_k, v_f_deriv_k, pol_f_k) where {M <: StochasticModel}
    error("not yet implemented")
    return hjb_err, pol_err
end


function monotonicity_penalty(outputs)
    penalty = 0.0
    for i in 2:length(outputs)
        # Assuming outputs are meant to be increasing
        penalty += max(0.0, outputs[i-1] - outputs[i])
    end
    return penalty
end
function calculate_lipschitz_constant(xs, ys)
    n = length(xs)
    max_ratio = 0.0
    
    for i in 1:n-1
        # Compute the difference ratio for adjacent pairs of points
        y_diff = abs(ys[i+1] - ys[i])
        x_diff = abs(xs[i+1] - xs[i])
        
        # Avoid division by zero
        if x_diff != 0
            ratio = y_diff / x_diff
            max_ratio = max(max_ratio, ratio)
        end
    end
    
    return max_ratio
end





function generate_model_values(model_name) 
    bounds_dict = Dict(
        "SkibaModel" => Dict(
            # γ, α, ρ, δ, A_H, A_L, κ
            # "ub" => [10.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0],
            # "lb" => [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            "ub" => [3.0, 0.9, 0.9, 0.5, 1.0, 1.0, 10.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "SmoothSkibaModel" => Dict(
            # γ, α, ρ, δ, A_H, A_L, κ, β 
            "ub" => [10.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 1e5],
            "lb" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "RamseyCassKoopmansModel" => Dict(
            # γ, α, ρ, δ, A
            "ub" => [10.0, 1.0, 1.0, 1.0, 20.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0]
    ),
        "StochasticSkibaModel" => Dict(
            # γ, α, ρ, δ, A_H, A_L, κ, θ, σ
            "ub" => [3.0, 0.9, 0.9, 0.5, 1.0, 1.0, 10.0, 1.0, 3.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
    )


    lb = bounds_dict[model_name]["lb"]
    ub = bounds_dict[model_name]["ub"]
    sobol_seq = SobolSeq(lb, ub)

    return sobol_seq
end

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

function f_nn(nn, k::AbstractVector, model_params, ps, st)
    return first(Lux.apply(nn, vcat(k', model_params), ps, st))
end

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



# finite diff accounting for jumps
# this one reuses v(k)
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
# finite diff, jumps, computes v(k)
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


function predict_fn(nets, k, model_params, nn_params, nn_states; derivative = true)
    v_nn, pol_nn = nets
    v_f_k = f_nn(v_nn, k, model_params, nn_params[1], nn_states[1]) |> vec
    if derivative
        v_f_deriv_k = f_nn_deriv(v_nn, v_f_k, k, model_params, nn_params[1], nn_states[1]) |> vec
    else
        v_f_deriv_k = nothing
    end
    pol_f_k = f_nn(pol_nn, k, model_params, nn_params[2], nn_states[2]) |> vec
    return v_f_k, v_f_deriv_k, pol_f_k
end



function plot_pred_output(k_vals, v_f_k, v_f_deriv_k, pol_f_k)
    p1 = plot(
        vec(k_vals),
        v_f_k,
        colour = :blue,
        label = "NN \$V(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$V(k)\$",
        )
    p2 = plot(
        vec(k_vals),
        v_f_deriv_k,
        colour = :blue,
        label = "NN \$V'(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$V'(k)\$",
        )
    p3 = plot(
        vec(k_vals),
        pol_f_k,
        colour = :blue,
        label = "NN \$c(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$c(k)\$",
        )
    return p1, p2, p3
end
function moving_average(data, window_size)
    # Calculate the moving average using a window of specified size
    filter_length = length(data) - window_size + 1
    ma = zeros(filter_length)
    for i in 1:filter_length
        ma[i] = sum(filter(!isfinite, filter(!isnan, data[i:i+window_size-1]))) / window_size
    end
    return ma
end



function average_last_n(vec::Vector, n::Int)
    # Check if n is within the bounds of the vector length
    if n > length(vec) || n < 1
        error("n must be between 1 and the length of the vector")
    end

    # Get the last n elements of the vector
    last_n_elements = vec[end-n+1:end]

    # Filter out Inf and NaN values
    filtered_elements = filter(x -> !isinf(x) && !isnan(x), last_n_elements)

    # If all elements are Inf or NaN, return Inf
    if isempty(filtered_elements)
        return Inf
    end

    # Calculate the average
    return median(filtered_elements) 
end



function plot_nn_output( 
                        nets, 
                        k_vals, 
                        param_vals, 
                        nn_params, 
                        states, 
                        epoch_list, 
                        loss_list, 
                        upwind_targets,
                        cpu_dev,
                        m::DeterministicModel
                        ) 
    upwind_k,  upwind_v, upwind_pol, upwind_kdot = upwind_targets

    cpu_k_vals = k_vals |> cpu_dev
    cpu_param_vals = param_vals |> cpu_dev
    v_f_k, v_f_deriv_k, pol_f_k = predict_fn(nets, k_vals, param_vals, nn_params, states) |> cpu_dev
    kdot = production_function(m, cpu_k_vals) .- cpu_param_vals[4, :] .* cpu_k_vals .- pol_f_k
    hjb_err, pol_err = err_HJB(cpu_k_vals, cpu_param_vals, v_f_k, v_f_deriv_k, pol_f_k)

    p1, p2, p3 = plot_pred_output(cpu_k_vals, v_f_k, v_f_deriv_k, pol_f_k)
    plot!(p1, Array(upwind_k), Array(upwind_v), linewidth = 2, colour = :red, label = "Upwind")
    plot!(p3, Array(upwind_k), Array(upwind_pol), linewidth = 2, colour = :red, label = "Upwind")

    p4 = plot(
        epoch_list, 
        loss_list, 
        label = "Loss", 
        yscale = :log10,
        alpha = 0.2
        )

    p5 = plot(
        cpu_k_vals, 
        kdot, 
        label = "",
        xlabel = "\$k\$",
        ylabel = "\$\\dot{k}\$",
        )
    plot!(p5, Array(upwind_k), Array(upwind_kdot), label = "Upwind", colour = :red, linewidth = 2)
    p6 = plot(
        cpu_k_vals, 
        hjb_err, 
        seriestype = :scatter,
        label = "",
        xlabel = "\$k\$",
        ylabel = "\$HJB Error\$",
        )
    p7 = plot(
        cpu_k_vals, 
        pol_err, 
        seriestype = :scatter,
        label = "",
        xlabel = "\$k\$",
        ylabel = "\$Policy Error\$",
        )
    return plot(p1, p2, p3, p4, p5, p6, p7, layout = (4, 2), size = (800, 800))
end

function plot_nn_output( 
                        nets, 
                        k_vals, 
                        param_vals, 
                        nn_params, 
                        states, 
                        epoch_list, 
                        loss_list, 
                        upwind_targets,
                        cpu_dev,
                        m::StochasticModel
                        ) 

    

    state_vals, upwind_v, upwind_pol, upwind_kdot = upwind_targets 
    v_f_k, _, pol_f_k = predict_fn(nets, state_vals, param_vals, nn_params, states, derivative = false) 

    v_kdot = production_function(m, state_vals[1, :], state_vals[2, :]) .- m.δ .* state_vals[1, :] .- pol_f_k 
    v_kdot = Array(v_kdot) |> cpu_dev
    u_states = Array(upwind_targets[1]) |> cpu_dev
    v = Array(v_f_k) |> cpu_dev
    pol = Array(pol_f_k) |> cpu_dev
    nk = length(unique(u_states[1, :]))
    nz = length(unique(u_states[2, :]))

    group = repeat(collect(1:nz), inner = nk)

    u_v = Array(upwind_targets[2])
    u_pol = Array(upwind_targets[3])
    u_kdot = Array(upwind_targets[4])

    p1 = plot(
        u_states[1, :], 
        v_kdot,
        group = group,
        colour = :blue,
        xlabel = "\$k\$",
        ylabel = "\$\\dot(k)\$",
        label = "",
        title = "\$\\dot{k}\$"
        )
    plot!(
        u_states[1, :],
        u_kdot,
        group = group,
        colour = :red,
        xlabel = "\$k\$",
        ylabel = "\$\\dot(k)\$",
        label = ""
    )
    
    p2 = plot(
        u_states[1, :], 
        v,
        group = group,
        colour = :blue,
        xlabel = "\$k\$",
        ylabel = "\$V(k)\$",
        title = "Value Function"
        )
   plot!(
    u_states[1, :],
    u_v,
        group = group,
        colour = :red,
        xlabel = "\$k\$",
        ylabel = "\$V(k)\$",
        label = ""
   ) 

   p3 = plot(
    u_states[1, :],
    pol,
    group = group,
    colour = :blue,
    xlabel = "\$k\$",
    ylabel = "\$c(k)\$",
    label = "",
    title = "Policy Function"
   )
   plot!(
    u_states[1, :],
    u_pol,
    group = group,
    colour = :red,
    xlabel = "\$k\$",
    ylabel = "\$c(k)\$",
    label = ""
   )

   mean_loss = round(average_last_n(loss_list, length(epoch_list) ÷ 10), digits = 3)

    p4 = plot(
        epoch_list, 
        loss_list, 
        label = "Loss", 
        yscale = :log10,
        alpha = 0.7,
        title = "Loss: $mean_loss"
        )
    ylims!(p4, minimum(loss_list), 1e4)

    return plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 800))
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



function create_upwind_targets(sm::SolvedModel{T}, res, model_params, device) where {T <: DeterministicModel}
    upwind_v = res.value.v |> device
    upwind_pol = sm.variables[:c] |> device
    upwind_kdot = sm.kdot_function(sm.variables[:k]) |> device
    upwind_model_params = repeat(model_params[:, 1], 1, size(upwind_v, 1))
    upwind_k = sm.variables[:k] |> device
    return  (upwind_k, upwind_v, upwind_pol, upwind_kdot), upwind_model_params
end

function create_upwind_targets(sm::SolvedModel{T}, res, model_params, device) where {T <: StochasticModel}
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


function check_gradients(grads)
    # Recursive function to check for NaN in gradients within any structure
    for grad in grads
        if grad isa NamedTuple && !haskey(grad, :weight)  # Check if the gradient component is a tuple (e.g., from Parallel)
            check_gradients(grad)  # Recurse into the tuple
        elseif grad isa NamedTuple && haskey(grad, :weight)  # Check if it's a layer with weights
            if any(isnan, grad.weight)
                return true  # Return true if any NaN is found
            end
        end
    end
    return false  # No NaN found
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
end