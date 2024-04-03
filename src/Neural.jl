using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using NaNMath


struct PositiveDense{F1, F2} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
end
function PositiveDense(in_dims::Int, out_dims::Int; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return PositiveDense{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight, init_bias)
end
function Lux.initialparameters(rng::AbstractRNG, layer::PositiveDense)
    w = layer.init_weight(rng, layer.out_dims, layer.in_dims)
    b = layer.init_bias(rng, layer.out_dims, 1)
    return (weight = w, bias = b)
end

Lux.initialstates(::AbstractRNG, ::PositiveDense) = NamedTuple()
Lux.parameterlength(l::PositiveDense) = l.out_dims * l.in_dims + l.out_dims
Lux.statelength(::PositiveDense) = 0


function (l::PositiveDense)(x::AbstractVecOrMat, ps, st::NamedTuple)
    # exp to ensure positive in input layer
    y = (exp.(ps.weight) * x) .+ ps.bias
    return y, st
end

abstract type GrowthModelLayer <: Lux.AbstractExplicitLayer end

struct MicawberLayer{F1, F2} <: GrowthModelLayer 
    m::Model
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

struct SteadyStateLayer{F1, F2} <: GrowthModelLayer
    m::Model
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

function MicawberLayer(in_dims::Int, out_dims::Int, m::Model; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return MicawberLayer{typeof(init_weight), typeof(init_bias)}(m, in_dims, out_dims, init_weight, init_bias)
end

function SteadyStateLayer(in_dims::Int, out_dims::Int, m::Model; init_weight=Lux.glorot_uniform, init_bias = Lux.zeros32)
    return SteadyStateLayer{typeof(init_weight), typeof(init_bias)}(m, in_dims, out_dims, init_weight, init_bias)
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
    return y, st
end

function (l::SteadyStateLayer)(x::AbstractVecOrMat, ps, st::NamedTuple)
    # exp to ensure positive in input layer
    # difference between x and k_star
    k_ss = k_steady_state(m)
    diff = -1 .* (k_ss .- x)
    abs_diff = abs.(diff)
    distances = [diff[argmin(abs_diff[:, x]), x] for x in axes(diff, 2)]';

    y = (exp.(ps.weight) * distances) .+ ps.bias
    return y, st
end





function err_HJB(k, model, v_f_k, v_f_deriv_k, pol_f_k)
    (; ρ, δ, γ) = model
    (; γ, α, ρ, δ, A_H, A_L, κ) = model
    θ = [γ, α, ρ, δ, A_H, A_L, κ]

    # c = max.(v_f_deriv_k, 1e-3) .^ (-1 / γ)
    # c = NaNMath.pow.(v_f_deriv_k, -1 / γ)

    # pos_deriv = v_f_deriv_k .> 0.0
    # v_f_k = v_f_k[pos_deriv]
    # v_f_deriv_k = v_f_deriv_k[pos_deriv]
    # pol_f_k = pol_f_k[pos_deriv]
    # k = k[pos_deriv]


    c = v_f_deriv_k .^ (-1 / γ)
    hjb_err = ρ .* v_f_k  .- (c .^ (1 - γ)) ./ (1 - γ) .- v_f_deriv_k .* (production_function(model, k) .- δ .* k .- pol_f_k)
    pol_err = c .- pol_f_k
    return hjb_err, pol_err
end



a = SteadyStateLayer(1, 256, m)
b = MicawberLayer(1, 256, m)




batch_size = 100
vals = reshape(collect(range(0.1, 2*k_star(m), length = batch_size)), (1, batch_size))

v_f_nn = Chain(
    # both read in simultaneously
    Parallel(
        nothing,
        SteadyStateLayer(1, 1, m),
        MicawberLayer(1, 1, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    # combine to 256*2 here
    PositiveDense(3, 256),
    PositiveDense(256, 1)
)




pol_f_nn = Chain(
    Dense(1 => 256, tanh),
    Dense(256 => 256, tanh),
    Dense(256 => 1, softplus),
)

rng = Random.default_rng()

vf_ps, vf_st = Lux.setup(rng, v_f_nn)
pol_ps, pol_st = Lux.setup(rng, pol_f_nn)

vf_y, vf_st = Lux.apply(v_f_nn, vals, vf_ps, vf_st)


v_f(k, ps, st) = Lux.apply(v_f_nn, k, ps, st)[1]
pol_f(k, ps, st) = Lux.apply(pol_f_nn, k, ps, st)[1]
# v_f_deriv(k, ps, st) = Zygote.forwarddiff(x -> v_f(x, ps, st), k)
v_f_deriv(k, ps, st) = Zygote.forwarddiff(z -> diag(ForwardDiff.jacobian(x -> v_f(x, ps, st), z)), k)


v_f_deriv(k, ps, st) = [Zygote.forwarddiff(z -> ForwardDiff.derivative(x -> v_f([x], ps, st)[1], z), y) for y in k]

v_f(vals, vf_ps, vf_st)
pol_f(vals, pol_ps, pol_st)
v_f_deriv(vals, vf_ps, vf_st)

plot(
    vals,
    v_f(vals, vf_ps, vf_st),
    seriestype = :scatter,
    label = "",
    colour = :black
)
plot(
    vals,
    v_f_deriv(vals, vf_ps, vf_st),
    seriestype = :scatter,
    label = "",
    colour = :red
)


using BenchmarkTools
@btime v_f(vals, vf_ps, vf_st);
@btime pol_f(vals, pol_ps, pol_st);
@btime v_f_deriv(vals, vf_ps, vf_st);


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
    hjb_err, pol_err = err_HJB(vec(x), m, v_f_k, v_f_deriv_k, pol_f_k)
    loss = sum(abs2, hjb_err) + sum(abs2, pol_err) + 1e10 * sum(v_f_deriv_k .< 0.0)
    return loss, vf_st, pol_st
end

# using BenchmarkTools
# @btime v_f(vals, vf_ps, vf_st);
# @btime pol_f(vals, vf_ps, vf_st);


v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, vf_ps, pol_ps, vf_st, pol_st)

loss_fn(vals, m, vf_ps, pol_ps, vf_st, pol_st)





for epoch in 1:500_000
    (loss, states...), back = Zygote.pullback(param_vec) do p
        loss_fn(vals, m, p.vf, p.pol, states[1], states[2])
    end
    grads = back((1.0, nothing, nothing))[1]
    epoch % 500 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")
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

h_err, p_err = err_HJB(vec(vals), m, sm_v_f_k, sm_v_f_deriv_k, sm_pol_f_k)

sum(abs2, h_err)
sum(abs2, p_err)
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
    #  label = "Policy Function",
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

?Zygote.pullback

param_vec.vf[2]

vf_ps[1]
Zygote.pullback(vf_ps) do 
    println("Hello")
end

vf_ps





(a, b), c = pullback(
    x -> Lux.apply(v_f_nn, x, ps, st),
    [0.2]
)

gs = pb((one.(l), nothing))

a
c((one.(a), nothing))[1]
a
b
c
