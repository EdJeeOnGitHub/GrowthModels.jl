module NeuralGrowthModel

using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxCUDA
using Plots
using Sobol

export PositiveDense, 
       MicawberLayer, 
       SteadyStateLayer, 
       err_HJB, 
       monotonicity_penalty, 
       calculate_lipschitz_constant, 
       generate_model_values, 
       generate_grid_values, 
       v_f, 
       pol_f, 
       dfx, 
       v_f_deriv, 
       predict_fn, 
       plot_pred_output, 
       plot_nn_output

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
    m::T
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

struct SteadyStateLayer{F1, F2, T <: Model} <: GrowthModelLayer
    activation
    m::T
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

function MicawberLayer(in_dims::Int, out_dims::Int, activation, m::Model; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return MicawberLayer{typeof(init_weight), typeof(init_bias), typeof(m)}(activation, m, in_dims, out_dims, init_weight, init_bias)
end


function SteadyStateLayer(in_dims::Int, out_dims::Int, activation, m::Model; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return SteadyStateLayer{typeof(init_weight), typeof(init_bias), typeof(m)}(activation, m, in_dims, out_dims, init_weight, init_bias)
end

function Lux.initialparameters(rng::AbstractRNG, layer::GrowthModelLayer)
    w = layer.init_weight(rng, layer.out_dims, 1)
    b = layer.init_bias(rng, layer.out_dims, 1)
    return (weight = w, bias = b)
end
Lux.initialstates(::AbstractRNG, ::GrowthModelLayer) = NamedTuple()
Lux.parameterlength(l::GrowthModelLayer) = l.out_dims * l.in_dims + l.out_dims

m = SkibaModel()

function (l::MicawberLayer{F1, F2, M})(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple) where {F1, F2, T <: Real, M <: SkibaModel}
    k = x[1, :]
    γ, α, ρ, δ, A_H, A_L, κ = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :], x[7, :], x[8, :]
    k_s = κ ./ (1 .- (A_L ./ A_H).^(1 ./ α))
    # abs to ensure positive in input layer
    # difference between x and k_star
    y = (ps.weight * (k .- k_s)') .+ ps.bias
    return l.activation.(y), st
end


k_steady_state_hi_Skiba(α::Real, A_H::Real, ρ::Real, δ::Real, κ::Real) = (α*A_H/(ρ + δ))^(1/(1-α)) + κ
k_steady_state_lo_Skiba(α::Real, A_L::Real, ρ::Real, δ::Real) = (α*A_L/(ρ + δ))^(1/(1-α))

function (l::SteadyStateLayer{F1, F2, M})(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple) where {F1, F2, T <: Real, M <: SkibaModel}
    k = x[1, :]
    γ, α, ρ, δ, A_H, A_L, κ = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :], x[7, :], x[8, :]
    # abs to ensure positive in input layer
    # difference between x and k_star
    k_ss_hi = (α .* A_H ./ (ρ .+ δ)) .^ (1 ./ (1 .- α)) .+ κ
    k_ss_lo = (α .* A_L ./ (ρ .+ δ)) .^ (1 ./ (1 .- α))
    # k_ss_hi = GrowthModels.k_steady_state_hi_Skiba.(α, A_H, ρ, δ, κ)
    # k_ss_lo = GrowthModels.k_steady_state_lo_Skiba.(α, A_L, ρ, δ)
    k_ss = [k_ss_lo; k_ss_hi] 
    diff = -1 *(k_ss .- k')

    abs_diff = abs.(diff)
    distances_indices = argmin(abs_diff, dims = 1)
    distances = diff[distances_indices]
    y = (ps.weight * distances) .+ ps.bias
    return l.activation.(y), st
end


function err_HJB(k, model_params, v_f_k, v_f_deriv_k, pol_f_k)
    non_neg_deriv_k = v_f_deriv_k .> 0


    v_f_k = v_f_k[non_neg_deriv_k]
    v_f_deriv_k = v_f_deriv_k[non_neg_deriv_k]
    pol_f_k = pol_f_k[non_neg_deriv_k]
    k = k[non_neg_deriv_k]

    γ, α, ρ, δ, A_H, A_L, κ = model_params[1, non_neg_deriv_k], model_params[2, non_neg_deriv_k], model_params[3, non_neg_deriv_k], model_params[4, non_neg_deriv_k], model_params[5, non_neg_deriv_k], model_params[6, non_neg_deriv_k], model_params[7, non_neg_deriv_k]
    # v_f_deriv_k = max.(v_f_deriv_k, Float32(1e-4))
    c = v_f_deriv_k .^ (-1 ./ γ)
    hjb_err = ρ .* v_f_k  .- (c .^ (1 .- γ)) ./ (1 .- γ) .- v_f_deriv_k .* (GrowthModels.skiba_production_function.(k, α, A_H, A_L, κ) .- δ .* k .- pol_f_k)
    pol_err = c .- pol_f_k
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
            "ub" => [3.0, 0.9, 0.9, 0.5, 1.0, 1.0, 5.0],
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
            "ub" => [10.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 1.0, 3.0],
            "lb" => [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
    )


    lb = bounds_dict[model_name]["lb"]
    ub = bounds_dict[model_name]["ub"]
    sobol_seq = SobolSeq(lb, ub)

    return sobol_seq
end

function generate_grid_values(m::Model, batch_size::Int; seed = 1234)
    Random.seed!(seed)
    k_ss = k_steady_state(m)
    # k_st = min(GrowthModels.k_star(m), maximum(k_ss))
    k_st = maximum(k_ss);
    # model_centers = sample([k_ss; k_st], batch_size ÷ 2, replace = true)
    # model_points = abs.(randn(length(model_centers)) .* 0.01 .+ model_centers)
    sob_seq = SobolSeq(1e-5, 1.3*maximum(k_ss))
    grid_points = reduce(hcat, next!(sob_seq) for i = 1:(batch_size ÷ 2))

    @show maximum(grid_points)
    # points = sort(vcat(model_points, grid_points[1, :]))
    points = sort(vcat(grid_points[1, :], grid_points[1, :]))

    m_vals = params(m)
    m_grid = repeat(m_vals, 1, batch_size)
    vals = vcat(points', m_grid)
    return Float32.(vals)
end


function v_f(nn, k, model_params, ps, st)
    return first(Lux.apply(nn, vcat(k', model_params), ps, st))

end
function v_f(nn, k, ps, st)
    return first(Lux.apply(nn, k, ps, st)) 
end

function pol_f(nn, k, model_params, ps, st)
    return first(Lux.apply(nn, vcat(k', model_params), ps, st)) 
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
function v_f_deriv(nn, v, k, model_params, ps, st; h = Float32(1e-3))
    fwd_v = v_f(nn, k .+ h, model_params, ps, st)
    bwd_v = v_f(nn, k .- h, model_params, ps, st)
    central_diff = (fwd_v .- bwd_v) ./ (2 * h)
    fwd_diff = (fwd_v .- v) ./ h
    bwd_diff = (v .- bwd_v) ./ h
    diffs = vcat(central_diff, bwd_diff, fwd_diff)
    abs_diffs = abs.(diffs)
    return diffs[argmin(abs_diffs, dims = 1)]
end
# finite diff, jumps, computes v(k)
function v_f_deriv(nn, k, model_params, ps, st; h = Float32(1e-3))
    v = v_f(nn, k, model_params, ps, st)
    fwd_v = v_f(nn, k .+ h, model_params, ps, st)
    bwd_v = v_f(nn, k .- h, model_params, ps, st)
    central_diff = (fwd_v .- bwd_v) ./ (2 * h)
    fwd_diff = (fwd_v .- v) ./ h
    bwd_diff = (v .- bwd_v) ./ h
    diffs = vcat(central_diff, bwd_diff, fwd_diff)
    abs_diffs = abs.(diffs)
    return diffs[argmin(abs_diffs, dims = 1)]
end




function predict_fn(fns, nets, k, model_params, vf_ps, pol_ps, vf_st, pol_st)
    v_f, v_f_deriv, pol_f = fns
    v_nn, pol_nn = nets
    v_f_k = v_f(v_nn, k, model_params, vf_ps, vf_st)
    v_f_deriv_k = v_f_deriv(v_nn, v_f_k, k, model_params, vf_ps, vf_st)
    pol_f_k = pol_f(pol_nn, k, model_params, pol_ps, pol_st)
    return  vec(v_f_k), vec(v_f_deriv_k), vec(pol_f_k)
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

function plot_nn_output(fns, 
                        nets, 
                        k_vals, 
                        param_vals, 
                        nn_params, 
                        states, 
                        epoch_list, 
                        loss_list, 
                        upwind_k,
                        upwind_v,
                        upwind_pol,
                        upwind_kdot,
                        cpu_dev) 
        cpu_k_vals = k_vals |> cpu_dev
        cpu_param_vals = param_vals |> cpu_dev
        v_f_k, v_f_deriv_k, pol_f_k = predict_fn(fns, nets, k_vals, param_vals, nn_params[1], nn_params[2], states[1], states[2]) |> cpu_dev
        kdot = production_function(m, cpu_k_vals) .- cpu_param_vals[4, :] .* cpu_k_vals .- pol_f_k
        hjb_err, pol_err = err_HJB(cpu_k_vals, cpu_param_vals, v_f_k, v_f_deriv_k, pol_f_k)


        p1, p2, p3 = plot_pred_output(cpu_k_vals, v_f_k, v_f_deriv_k, pol_f_k)
        plot!(p1, Array(upwind_k), Array(upwind_v), linewidth = 2, colour = :red, label = "Upwind")
        plot!(p3, Array(upwind_k), Array(upwind_pol), linewidth = 2, colour = :red, label = "Upwind")

        p4 = plot(
            epoch_list, 
            loss_list, 
            label = "Loss", 
            yscale = :log10
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

end