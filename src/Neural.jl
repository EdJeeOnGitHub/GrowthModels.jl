using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxCUDA
using Plots
using Sobol


benchmark = false
CUDA.allowscalar(true)

Random.seed!(1234)
cpu_dev = cpu_device()
device = gpu_device()
# Stuff for GPU
@inline GrowthModels.production_function(::SkibaModel, k::Union{Real,AbstractArray}, α::Real, A_H::Real, A_L::Real, κ::Real) = skiba_production_function(k, α, A_H, A_L, κ)
@inline GrowthModels.production_function(::SkibaModel, k::Union{Real,AbstractArray}, params::Vector) = skiba_production_function.(k, params[1], params[2], params[3], params[4])
@inline GrowthModels.production_function(m::SkibaModel, k::Union{Real,AbstractArray}) = skiba_production_function.(k, m.α, m.A_H, m.A_L, m.κ)
GrowthModels.k_steady_state(m, device::Lux.AbstractLuxDevice) = device(k_steady_state(m))
function SkibaModel{T}(; γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0) where {T<: Real}
    SkibaModel{T}(γ, α, ρ, δ, A_H, A_L, κ)
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
    k_ss = [k_ss_lo; k_ss_hi] |> device
    diff = -1.0f32 *(k_ss .- k')

    abs_diff = abs.(diff)
    distances_indices = argmin(abs_diff, dims = 1)
    distances = diff[distances_indices]
    y = (ps.weight * distances) .+ ps.bias
    return l.activation.(y), st
end


function err_HJB(k, model_params, v_f_k, v_f_deriv_k, pol_f_k)
    # non_neg_deriv_k = v_f_deriv_k .> 0

    # v_f_k = v_f_k[non_neg_deriv_k]
    # v_f_deriv_k = v_f_deriv_k[non_neg_deriv_k]
    # pol_f_k = pol_f_k[non_neg_deriv_k]
    # k = k[non_neg_deriv_k]

    γ, α, ρ, δ, A_H, A_L, κ = model_params[1, :], model_params[2, :], model_params[3, :], model_params[4, :], model_params[5, :], model_params[6, :], model_params[7, :]
    v_f_deriv_k = max.(v_f_deriv_k, 0.1)
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



m = SkibaModel{Float32}() |> device
model_params = Float32.(params(m))
n_params = length(model_params)

batch_size = 500


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
    k_st = GrowthModels.k_star(m)
    model_centers = sample([k_ss; k_st], batch_size ÷ 2, replace = true)
    model_points = abs.(randn(Float32, length(model_centers)) .* 0.01 .+ model_centers)
    sob_seq = SobolSeq(1e-5, 2*maximum(k_ss))
    grid_points = reduce(hcat, next!(sob_seq) for i = 1:(batch_size ÷ 2))
    points = sort(vcat(model_points, grid_points[1, :]))

    m_vals = params(m)
    m_grid = repeat(m_vals, 1, batch_size)
    vals = vcat(points', m_grid)
    return Float32.(vals) 
end

vals = generate_grid_values(m, batch_size)
cpu_vals = deepcopy(vals)
cpu_k_vals = vec(cpu_vals[1, :])
cpu_param_vals = cpu_vals[2:end, :]
vals = vals |> device
k_vals = vec(vals[1, :])
param_vals = vals[2:end, :]
c_size = 10
n_size = 10

v_f_nn = Chain(
    # both read in simultaneously
    Parallel(
        nothing,
        MicawberLayer(1 + n_params, c_size, relu, m),
        SteadyStateLayer(1 + n_params, c_size, relu, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    Dense(c_size*2 + 1 + n_params => n_size, relu),
    Dense(n_size => n_size, relu),
    Dense(n_size => 1)
)


pol_f_nn = Chain(
    Parallel(
        nothing,
        SteadyStateLayer(1 + n_params, c_size, relu, m),
        MicawberLayer(1 + n_params, c_size, relu, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    Dense(c_size*2 + 1 + n_params => n_size, relu),
    Dense(n_size => n_size, relu),
    Dense(n_size => 1, softplus),
)

rng = Random.default_rng()
cpu_vf_ps, cpu_vf_st = Lux.setup(rng, v_f_nn) 
cpu_pol_ps, cpu_pol_st = Lux.setup(rng, pol_f_nn) 

vf_ps, vf_st = Lux.setup(rng, v_f_nn) .|> device
pol_ps, pol_st = Lux.setup(rng, pol_f_nn) .|> device

vf_y, vf_st = Lux.apply(v_f_nn, vals, vf_ps, vf_st)


function v_f(k, model_params, ps, st)
    return first(Lux.apply(v_f_nn, vcat(k', model_params), ps, st))

end
function v_f(k, ps, st)
    return first(Lux.apply(v_f_nn, k, ps, st)) 
end

function pol_f(k, model_params, ps, st)
    return first(Lux.apply(pol_f_nn, vcat(k', model_params), ps, st)) 
end


# finite diff accounting for jumps
# this one reuses v(k)
function v_f_deriv(v, k, model_params, ps, st; h = Float32(1e-3))
    fwd_v = v_f(k .+ h, model_params, ps, st)
    bwd_v = v_f(k .- h, model_params, ps, st)
    central_diff = (fwd_v .- bwd_v) ./ (2 * h)
    fwd_diff = (fwd_v .- v) ./ h
    bwd_diff = (v .- bwd_v) ./ h
    diffs = vcat(central_diff, bwd_diff, fwd_diff)
    abs_diffs = abs.(diffs)
    return diffs[argmin(abs_diffs, dims = 1)]
end
# finite diff, jumps, computes v(k)
function v_f_deriv(k, model_params, ps, st; h = Float32(1e-3))
    v = v_f(k, model_params, ps, st)
    fwd_v = v_f(k .+ h, model_params, ps, st)
    bwd_v = v_f(k .- h, model_params, ps, st)
    central_diff = (fwd_v .- bwd_v) ./ (2 * h)
    fwd_diff = (fwd_v .- v) ./ h
    bwd_diff = (v .- bwd_v) ./ h
    diffs = vcat(central_diff, bwd_diff, fwd_diff)
    abs_diffs = abs.(diffs)
    return diffs[argmin(abs_diffs, dims = 1)]
end



using BenchmarkTools
if benchmark
    @btime v_f(k_vals, param_vals, vf_ps, vf_st);
    @btime v_f(vals, vf_ps, vf_st);
    @btime v_f_deriv(k_vals, param_vals, vf_ps, vf_st);
end


l, b = Zygote.pullback(vf_ps) do p
    v_f_deriv(k_vals, param_vals, p, vf_st)
end;

if benchmark
    @btime l, b = Zygote.pullback(vf_ps) do p
        v_f_deriv(k_vals, param_vals, p, vf_st)
    end;
end

rng = Random.default_rng()
cpu_vf_ps, cpu_vf_st = Lux.setup(rng, v_f_nn) 
cpu_pol_ps, cpu_pol_st = Lux.setup(rng, pol_f_nn) 

vf_ps, vf_st = Lux.setup(rng, v_f_nn) .|> device
pol_ps, pol_st = Lux.setup(rng, pol_f_nn) .|> device

vf_y, vf_st = Lux.apply(v_f_nn, vals, vf_ps, vf_st)


states = (vf_st, pol_st) |> device
nn_params = (vf_ps, pol_ps) |> device
opt = Optimisers.ADAM() 

st_opt = Optimisers.setup(ADAM(), nn_params) |> device



function predict_fn(k, model_params, vf_ps, pol_ps, vf_st, pol_st)
    v_f_k = v_f(k, model_params, vf_ps, vf_st)
    v_f_deriv_k = v_f_deriv(v_f_k, k, model_params, vf_ps, vf_st)
    pol_f_k = pol_f(k, model_params, pol_ps, pol_st)
    return  vec(v_f_k), vec(v_f_deriv_k), vec(pol_f_k)
end

function loss_fn(k, model_params, vf_ps, pol_ps, vf_st, pol_st)
    v_f_k, v_f_deriv_k, pol_f_k = predict_fn(k, model_params, vf_ps, pol_ps, vf_st, pol_st)
    
    hjb_err, pol_err = err_HJB(k, model_params, v_f_k, v_f_deriv_k, pol_f_k)

    neg_deriv_cost = max.(0, -v_f_deriv_k)

    n_k = length(k)
    loss = sqrt(sum(abs2, hjb_err) / n_k)  + 
            sqrt(sum(abs2, pol_err)  / n_k) +
            sqrt(sum(abs2, neg_deriv_cost) / n_k)
    # enforce k > 0
    # loss += sum(abs2, 1e3 .* (kt1 .< 0))
    # enforce value function monotonic
    # loss += monotonicity_penalty(v_f_k) 
    # lipschitz cost
    # loss += calculate_lipschitz_constant(vec_x, v_f_k) 
    return loss
end




# using BenchmarkTools
# @btime v_f(vals, vf_ps, vf_st);
# @btime pol_f(vals, vf_ps, vf_st);
v_f_k, v_f_deriv_k, pol_f_k = predict_fn(k_vals, param_vals, vf_ps, pol_ps, vf_st, pol_st)


function plot_pred_output(k_vals, v_f_k, v_f_deriv_k, pol_f_k)
    p1 = plot(
        vec(k_vals),
        v_f_k,
        seriestype = :scatter,
        colour = :blue,
        label = "NN \$V(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$V(k)\$",
        )
    p2 = plot(
        vec(k_vals),
        v_f_deriv_k,
        seriestype = :scatter,
        colour = :blue,
        label = "NN \$V'(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$V'(k)\$",
        )
    p3 = plot(
        vec(k_vals),
        pol_f_k,
        seriestype = :scatter,
        colour = :blue,
        label = "NN \$c(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$c(k)\$",
        )
    return p1, p2, p3
end

function plot_nn_output(k_vals, param_vals, nn_params, states, epoch_list, loss_list, cpu_dev) 
        cpu_k_vals = k_vals |> cpu_dev
        cpu_param_vals = param_vals |> cpu_dev
        v_f_k, v_f_deriv_k, pol_f_k = predict_fn(k_vals, param_vals, nn_params[1], nn_params[2], states[1], states[2]) |> cpu_dev
        kdot = production_function(m, cpu_k_vals) .- cpu_param_vals[4, :] .* cpu_k_vals .- pol_f_k
        hjb_err, pol_err = err_HJB(cpu_k_vals, cpu_param_vals, v_f_k, v_f_deriv_k, pol_f_k)


        p1, p2, p3 = plot_pred_output(cpu_k_vals, v_f_k, v_f_deriv_k, pol_f_k)
        p4 = plot(
            epoch_list, 
            loss_list, 
            label = "Loss", 
            seriestype = :scatter,
            yscale = :log10
            )
        p5 = plot(
            cpu_k_vals, 
            kdot, 
            seriestype = :scatter,
            label = "",
            xlabel = "\$k\$",
            ylabel = "\$\\dot{k}\$",
            )
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
# p1, p2, p3 = plot_pred_output(k_vals, v_f_k, v_f_deriv_k, pol_f_k)
loss_fn(k_vals, param_vals , vf_ps, pol_ps, vf_st, pol_st)

k = k_vals
model_params = param_vals
    v_f_k, v_f_deriv_k, pol_f_k = predict_fn(k, model_params, vf_ps, pol_ps, vf_st, pol_st)
    
    hjb_err, pol_err = err_HJB(k, model_params, v_f_k, v_f_deriv_k, pol_f_k)

    neg_deriv_cost = max.(0, -v_f_deriv_k)

    n_k = length(k)
    loss = sqrt(sum(abs2, hjb_err) / n_k)  + 
            sqrt(sum(abs2, pol_err)  / n_k) +
            sqrt(sum(abs2, neg_deriv_cost) / n_k)


        plot(k, v_f_deriv_k)

        plot(k, hjb_err)


if benchmark
    @btime l, back = Zygote.pullback(nn_params) do p
        loss_fn(k_vals, param_vals, p[1], p[2], states[1], states[2])
    end;
end


l, back = Zygote.pullback(nn_params) do p
    loss_fn(k_vals, param_vals, p[1], p[2], states[1], states[2])
end;

l


rng = Random.default_rng()
cpu_vf_ps, cpu_vf_st = Lux.setup(rng, v_f_nn) 
cpu_pol_ps, cpu_pol_st = Lux.setup(rng, pol_f_nn) 

vf_ps, vf_st = Lux.setup(rng, v_f_nn) .|> device
pol_ps, pol_st = Lux.setup(rng, pol_f_nn) .|> device

vf_y, vf_st = Lux.apply(v_f_nn, vals, vf_ps, vf_st)


states = (vf_st, pol_st) |> device
nn_params = (vf_ps, pol_ps) |> device
opt = Optimisers.ADAM() 

st_opt = Optimisers.setup(ADAM(), nn_params) |> device

skiba_sobol_seq = generate_model_values("SkibaModel")

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

    
epoch_list = [1]
loss_list = [Inf]

l, back = Zygote.pullback(nn_params) do p
    loss_fn(k_vals, param_vals, p[1], p[2], states[1], states[2])
end;






n_redraw = 1_000_000
for epoch in epoch_list[end]:1_000_000
# epoch = 1
# begin
    # epoch += 1
# grads[1][1][1].weight .= 0.0
# while !any(isnan, grads[1][1][1].weight)
    # epoch = epoch_list[end] + 1

    # if epoch % n_redraw == 0 || epoch == 1
    #     m = SkibaModel{Float32}(param_reshuffle(next!(skiba_sobol_seq))...) 
    #     max_ss = maximum(k_steady_state(m))
    #     state_constraint = check_statespace(m)
    #     while !(max_ss < 1e2) && !state_constraint
    #         model_param_candidate = param_reshuffle(next!(skiba_sobol_seq))
    #         m = SkibaModel{Float32}(model_param_candidate...) 
    #         max_ss = maximum(k_steady_state(m))
    #     end
    #     # m = m |> device
    #     m = SkibaModel()
    # end
    m = SkibaModel()
    random_vals = generate_grid_values(m, batch_size, seed = epoch) 
    k_vals = random_vals[1, :]
    param_vals = random_vals[2:end, :]
    cpu_k_vals = deepcopy(k_vals)
    cpu_param_vals = deepcopy(param_vals)
    k_vals = k_vals |> device
    param_vals = param_vals |> device

    # v_f_k, v_f_deriv_k, pol_f_k = predict_fn(k_vals, param_vals, nn_params[1], nn_params[2], states[1], states[2])

    # v_l, v_b = Zygote.pullback(nn_params[1]) do p
    #     v_f(k_vals, param_vals, Zygote.@showgrad(p), vf_st)
    # end;

    # pol_l, pol_b = Zygote.pullback(nn_params[2]) do p
    #     pol_f(k_vals, param_vals, p, pol_st)
    # end;

    # deriv_l, deriv_b = Zygote.pullback(nn_params[1]) do p
    #     v_f_deriv(k_vals, param_vals, p, vf_st)
    # end;

    # v_l
    # deriv_l
    # pol_l

    

    loss, back = Zygote.pullback(nn_params) do p
        loss_fn(k_vals, param_vals, p[1], p[2], states[1], states[2])
    end;

    if epoch % 10 == 1
        println("Epoch: $epoch, Loss: $loss")
    end
    push!(epoch_list, epoch)
    push!(loss_list, loss)
    if epoch % 500 == 1
        try 
            p_model_output = plot_nn_output(k_vals, param_vals, nn_params, states, epoch_list, loss_list, cpu_dev)
            display(p_model_output)
        catch e 
            println(e)
        end
    end
    if !isnan(loss)
        grads = back(1.0)[1]
        if any(isnan, grads[1][1][1].weight)
            println("NaN Gradients")
        else
            Optimisers.update!(st_opt, nn_params, grads)
        end
    end;
end;

random_vals
m

savefig("skiba-nn-fit.pdf")



v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, param_vec.vf, param_vec.pol, states[1], states[2])


skiba_hyperparams = StateSpaceHyperParams(m)
skiba_state = StateSpace(m, skiba_hyperparams)
skiba_init_value = Value(skiba_state);

fit_value, fit_variables, fit_iter = solve_HJB(
    m, 
    skiba_hyperparams, 
    init_value = skiba_init_value, maxit = 1000);

sm = SolvedModel(m, fit_value, fit_variables)

using DataInterpolations

sm_v_interp = DataInterpolations.LinearInterpolation(fit_value.v, fit_variables.k)
sm_v_deriv_interp = DataInterpolations.LinearInterpolation(fit_value.dVf, fit_variables.k)

sm_v_f_k = sm_v_interp(vec(vals))
sm_v_f_deriv_k = sm_v_deriv_interp(vec(vals))
sm_pol_f_k = sm.policy_function(vec(vals))

sm_h_err, sm_p_err = err_HJB(vec(vals), m, sm_v_f_k, sm_v_f_deriv_k, sm_pol_f_k)

nn_h_err, nn_p_err = err_HJB(vec(vals), m, v_f_k, v_f_deriv_k, pol_f_k)

sum(abs2, sm_h_err)
sum(abs2, sm_p_err)

sum(abs2, nn_h_err)
sum(abs2, nn_p_err)

v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, param_vec.vf, param_vec.pol, states[1], states[2])
kdot = production_function(m, vec_vals) .- m.δ .* vec_vals .- pol_f_k
hjb_err, pol_err = err_HJB(vec(vals), m, v_f_k, v_f_deriv_k, pol_f_k)


p1, p2, p3 = plot_pred_output(vals, v_f_k, v_f_deriv_k, pol_f_k)
## Adding upwind solution for comparison
plot!(
    p1, 
    vec_vals, 
    sm_v_f_k, 
    seriestype = :scatter, 
    label = "Upwind \$V(k)\$", 
    colour = :red)
plot!(
    p2,
    vec_vals,
    sm_v_f_deriv_k,
    seriestype = :scatter,
    label = "Upwind \$V'(k)\$",
    colour = :red
)
plot!(
    p3,
    vec_vals,
    sm_pol_f_k,
    seriestype = :scatter,
    label = "Upwind \$c(k)\$",
    colour = :red)

p4 = plot(
    epoch_list, 
    loss_list, 
    label = "Loss", 
    seriestype = :scatter,
    yscale = :log10,
    ylabel = "MSE",
    xlabel = "Epochs"
    )
p5 = plot(
    vec_vals, 
    kdot, 
    seriestype = :scatter,
    label = "NN \$\\dot{k}\$",
    xlabel = "\$k\$",
    ylabel = "\$\\dot{k}\$",
    )
plot!(
    p5,
    vec_vals,
    sm.kdot_function(vec_vals),
    seriestype = :scatter,
    label = "Upwind \$\\dot{k}\$",
    xlabel = "\$k\$",
    ylabel = "\$\\dot{k}\$"
)

p6 = plot(
    vec_vals, 
    hjb_err, 
    seriestype = :scatter,
    label = "",
    xlabel = "\$k\$",
    ylabel = "\$HJB Error\$",
    )
p7 = plot(
    vec_vals, 
    pol_err, 
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
    @btime loss_fn([10.0], m, param_vec.vf, param_vec.pol, states[1], states[2])
end
loss_fn(x) = loss_fn([x], m, vf_ps, pol_ps, vf_st, pol_st)

loss_fn.([0.2, 0.1])


for epoch in 1:2
    global vf_st, pol_st
    (loss, vf_st, pol_st), pb = Zygote.pullback(vf_ps) do p
        v_f_k, vf_st_ = Lux.apply(v_f_nn, [10.0], p, vf_st)
        pol_f_k, pol_st_ = Lux.apply(pol_f_nn, [10.0], pol_ps, pol_st)
        v_f_deriv_k = v_f_deriv([10.0], p, vf_st) 
        hjb_err, pol_err = err_HJB([10.0], m, v_f_k, v_f_deriv_k, pol_f_k)
        loss = sum(hjb_err .^ 2) + sum(pol_err .^ 2)
        return loss, vf_st_, pol_st_
    end 
    gs = only(pb((one(loss), nothing, nothing)))
    epoch % 100 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")
    Optimisers.update!(vf_opt, vf_ps, gs)
end


v_f_deriv([10.0], vf_ps, vf_st)

pullback(v_f_deriv, [10.0], vf_ps, vf_st)

pb((one(loss), vf_st, pol_st))




using Enzyme

x  = [2.0]
bx = [0.0]
y  = [0.0]

x = fill(2.0, (1, 2))
bx = fill(0.0, (1, 2))
x = [2.0 2.0]
bx = [0.0 0.0]

using ComponentArrays, Lux, Random

rng = Random.default_rng()
Random.seed!(rng,100)
dudt2 = Lux.Chain(x -> x.^3,
                  Lux.Dense(1, 50, tanh),
                  Lux.Dense(50, 1))
p, st = Lux.setup(rng, dudt2)

function f(x::Array{Float64})
    return dudt2(x, p, st)[1]
end

f(x)

Enzyme.autodiff(
    Reverse, 
    f, 
    Active,
    Duplicated([1.0 0.2], [0.0 0.0]))

Enzyme.autodiff(
    Reverse, 
    f, 
    Active,
    Duplicated(x, bx))

    x
    bx
bx

function f2(x::Array{Float64})
    dudt2(x, p, st)[1]
end
f2(x)
using Zygote
bx2 = Zygote.pullback(f2, x)[2](ones(size(x)))[1]

bx2([1.0 1.0])[1]
bx
bx2

@show bx - bx2

#=
2-element Vector{Float64}:
 -9.992007221626409e-16
 -1.7763568394002505e-15
=#