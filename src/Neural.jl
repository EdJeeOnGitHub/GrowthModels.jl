using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxAMDGPU 
using Plots

device = cpu_device()
# Stuff for GPU
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
    # exp to ensure positive in input layer
    y = (exp.(ps.weight) * x) .+ ps.bias
    return l.activation.(y), st
end

abstract type GrowthModelLayer <: Lux.AbstractExplicitLayer end

struct MicawberLayer{F1, F2} <: GrowthModelLayer 
    activation
    m::Model
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

struct SteadyStateLayer{F1, F2} <: GrowthModelLayer
    activation
    m::Model
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

function MicawberLayer(in_dims::Int, out_dims::Int, activation, m::Model; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return MicawberLayer{typeof(init_weight), typeof(init_bias)}(activation, m, in_dims, out_dims, init_weight, init_bias)
end

function SteadyStateLayer(in_dims::Int, out_dims::Int, activation, m::Model; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return SteadyStateLayer{typeof(init_weight), typeof(init_bias)}(activation, m, in_dims, out_dims, init_weight, init_bias)
end

function Lux.initialparameters(rng::AbstractRNG, layer::GrowthModelLayer)
    w = layer.init_weight(rng, layer.out_dims, layer.in_dims)
    b = layer.init_bias(rng, layer.out_dims, 1)
    return (weight = w, bias = b)
end
Lux.initialstates(::AbstractRNG, ::GrowthModelLayer) = NamedTuple()
Lux.parameterlength(l::GrowthModelLayer) = l.out_dims * l.in_dims + l.out_dims

function (l::MicawberLayer)(x::AbstractVecOrMat, ps, st::NamedTuple)
    # exp to ensure positive in input layer
    # difference between x and k_star
    y = (exp.(ps.weight) * (x .- k_star(m))) .+ ps.bias
    return l.activation.(y), st
end

function (l::SteadyStateLayer)(x::AbstractVecOrMat, ps, st::NamedTuple)
    # exp to ensure positive in input layer
    # difference between x and k_star
    k_ss = k_steady_state(m, device)
    diff = -1.0f32 .* (k_ss .- x)
    abs_diff = abs.(diff)
    distances = [diff[argmin(abs_diff[:, x]), x] for x in axes(diff, 2)]';

    y = (exp.(ps.weight) * distances) .+ ps.bias
    return l.activation.(y), st
end

function err_HJB(k, model, v_f_k, v_f_deriv_k, pol_f_k)
    (; ρ, δ, γ) = model
    (; γ, α, ρ, δ, A_H, A_L, κ) = model
    # θ = [γ, α, ρ, δ, A_H, A_L, κ]

    c = v_f_deriv_k .^ (-1 / γ)
    hjb_err = ρ .* v_f_k  .- (c .^ (1 - γ)) ./ (1 - γ) .- v_f_deriv_k .* (production_function(m, k) .- δ .* k .- pol_f_k)
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
batch_size = 100
# vals = randn(1, batch_size) .+ k_star(m)


grid_vals = reshape(collect(range(0.1, 2*k_star(m), length = batch_size ÷ 4)), (1, batch_size ÷ 4))
ss_lo_vals = randn(1, batch_size ÷ 4) .+ k_steady_state_lo(m)
ss_hi_vals = randn(1, batch_size ÷ 4) .+ k_steady_state_hi(m)
ss_star = randn(1, batch_size ÷ 4) .+ k_star(m)
vals = hcat(grid_vals, ss_lo_vals, ss_hi_vals, ss_star) 
vals = abs.(vals)
cpu_vals = sort(vals, dims = 1)
vals = sort(vals, dims = 1) |> device
vec_vals = vec(vals)
c_size = 12
v_f_nn = Chain(
    # both read in simultaneously
    Parallel(
        nothing,
        SteadyStateLayer(1, c_size, tanh, m),
        MicawberLayer(1, c_size, tanh, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    PositiveDense(c_size*2 + 1, 48),
    PositiveDense(48, 1)
)

pol_f_nn = Chain(
    Parallel(
        nothing,
        SteadyStateLayer(1, c_size, tanh, m),
        MicawberLayer(1, c_size, tanh, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    Dense(c_size*2 + 1 => 48, tanh),
    # Dense(100 => 100, tanh),
    Dense(48 => 1, softplus),
)

rng = Random.default_rng()
vf_ps, vf_st = Lux.setup(rng, v_f_nn) .|> device
pol_ps, pol_st = Lux.setup(rng, pol_f_nn) .|> device

vf_y, vf_st = Lux.apply(v_f_nn, vals, vf_ps, vf_st)




k_steady_state(m, device) .- vals

m_cpu = SkibaModel{Float32}()
k_ss = k_steady_state(m_cpu) |> device
vals .- k_ss
k_steady_state(m_cpu) 
cpu_vals .- k_steady_state(m_cpu)
k_steady_state_hi(m) .- vals
k_steady_state(m) 

v_f(k, ps, st) = Lux.apply(v_f_nn, k, ps, st)[1]
pol_f(k, ps, st) = Lux.apply(pol_f_nn, k, ps, st)[1]
v_f_deriv(k, ps, st) = [Zygote.forwarddiff(z -> ForwardDiff.derivative(x -> v_f([x], ps, st)[1], z), y) for y in k]


v_f(vals, vf_ps, vf_st)
pol_f(vals, pol_ps, pol_st)
v_f_deriv(vals, vf_ps, vf_st)
plot(
    vec_vals,
    v_f(vals, vf_ps, vf_st)',
    seriestype = :scatter,
    label = "",
    colour = :black
)
plot(
    vec_vals,
    v_f_deriv(vals, vf_ps, vf_st)',
    seriestype = :scatter,
    label = "",
    colour = :red
)


using BenchmarkTools
# @btime v_f(vals, vf_ps, vf_st);
# @btime pol_f(vals, pol_ps, pol_st);
# @btime v_f_deriv(vals, vf_ps, vf_st);


param_vec = ComponentArray(vf = vf_ps, pol = pol_ps)
states = (vf_st, pol_st)
opt = Optimisers.ADAM()

st_opt = Optimisers.setup(ADAM(), param_vec)



function predict_fn(x, vf_ps, pol_ps, vf_st, pol_st)
    v_f_k = v_f(x, vf_ps, vf_st)
    v_f_deriv_k = v_f_deriv(x, vf_ps, vf_st)
    pol_f_k = pol_f(x, pol_ps, pol_st)
    return  vec(v_f_k), vec(v_f_deriv_k), vec(pol_f_k)
end

function loss_fn(x, m, vf_ps, pol_ps, vf_st, pol_st)
    v_f_k, v_f_deriv_k, pol_f_k = predict_fn(x, vf_ps, pol_ps, vf_st, pol_st)
    vec_x = vec(x)
    hjb_err, pol_err = err_HJB(vec_x, m, v_f_k, v_f_deriv_k, pol_f_k)

    kdot = production_function(m, vec_x) .- m.δ .* vec_x .- pol_f_k
    kt1 = vec_x .+ kdot
    loss = sum(abs2, hjb_err) + sum(abs2, pol_err) 
    # enforce k > 0
    loss += sum(abs2, 100*(kt1 .< 0))
    # enforce value function monotonic
    loss += monotonicity_penalty(v_f_k) * 1e4
    return loss, vf_st, pol_st
end





# using BenchmarkTools
# @btime v_f(vals, vf_ps, vf_st);
# @btime pol_f(vals, vf_ps, vf_st);
v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, vf_ps, pol_ps, vf_st, pol_st)



function plot_pred_output(vals, v_f_k, v_f_deriv_k, pol_f_k)
    p1 = plot(
        vec(vals),
        v_f_k,
        seriestype = :scatter,
        label = "",
        xlabel = "\$k\$",
        ylabel = "\$Vf(k)\$",
        )
    p2 = plot(
        vec(vals),
        v_f_deriv_k,
        seriestype = :scatter,
        label = "",
        xlabel = "\$k\$",
        ylabel = "\$Vf'(k)\$",
        )
    p3 = plot(
        vec(vals),
        pol_f_k,
        seriestype = :scatter,
        label = "",
        xlabel = "\$k\$",
        ylabel = "\$c(k)\$",
        )
    return p1, p2, p3
end

plot_pred_output(vals, v_f_k, v_f_deriv_k, pol_f_k)
loss_fn(vals, m, vf_ps, pol_ps, vf_st, pol_st)



epoch_list = []
loss_list = []
for epoch in 1:1_000_000
    (loss, states...), back = Zygote.pullback(param_vec) do p
        loss_fn(vals, m, p.vf, p.pol, states[1], states[2])
    end
    grads = back((1.0, nothing, nothing))[1]
    epoch % 500 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")

    if epoch % 500 == 1
        push!(epoch_list, epoch)
        push!(loss_list, loss)
        v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, param_vec.vf, param_vec.pol, states[1], states[2])
        kdot = production_function(m, vec_vals) .- m.δ .* vec_vals .- pol_f_k

        p1, p2, p3 = plot_pred_output(vals, v_f_k, v_f_deriv_k, pol_f_k)
        p4 = plot(
            epoch_list, 
            loss_list, 
            label = "Loss", 
            seriestype = :scatter,
            yscale = :log10
            )
        p5 = plot(
            vec_vals, 
            kdot, 
            seriestype = :scatter,
            label = "",
            xlabel = "\$k\$",
            ylabel = "\$\\dot{k}\$",
            )
        p_all = plot(p1, p2, p3, p4, p5, layout = (3, 2), size = (800, 800))
        display(p_all)
    end
    # println("Epoch: $(epoch) | Loss: $(loss)")
    Optimisers.update!(st_opt, param_vec, grads)
end




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

vec_vals = vec(vals)

plot(
    vec_vals,
    sm_v_f_k,
    seriestype = :scatter,
    label =""
)

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
@btime loss_fn([10.0], m, param_vec.vf, param_vec.pol, states[1], states[2])

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

