using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote



function err_HJB(k, model, v_f_k, v_f_deriv_k, pol_f_k)
    (; ρ, δ, γ) = model
    (; γ, α, ρ, δ, A_H, A_L, κ) = model
    θ = [γ, α, ρ, δ, A_H, A_L, κ]
    c = v_f_deriv_k .^ (-1 / γ)
    hjb_err = ρ .* v_f_k  .- (c .^ (1 - γ)) ./ (1 - γ) .- v_f_deriv_k .* (production_function(model, k) .- δ .* k .- pol_f_k)
    pol_err = v_f_deriv_k .^(-1/γ) .- pol_f_k
    return hjb_err, pol_err
end


m = SkibaModel()

params = ComponentArray(nn = [1, 2], θ = [1, 2, 3, 4, 5, 6, 7])

v_f_nn = Chain(
    Dense(1 => 50, tanh),
    Dense(50 => 1, tanh)
)

pol_f_nn = Chain(
    Dense(1 => 50, tanh),
    Dense(50 => 1, softplus),
)

rng = Random.default_rng()

vf_ps, vf_st = Lux.setup(rng, v_f_nn)
pol_ps, pol_st = Lux.setup(rng, pol_f_nn)

vf_y, vf_st = Lux.apply(v_f_nn, [0.2], vf_ps, vf_st)

v_f(k, ps, st) = Lux.apply(v_f_nn, k, ps, st)[1]
pol_f(k, ps, st) = Lux.apply(pol_f_nn, k, ps, st)[1]
v_f_deriv(k, ps, st) = Zygote.jacobian(x -> v_f(x, ps, st)[1], k)[1][1]

v_f_deriv(k, ps, st) = Zygote.gradient(x -> first(v_f(x, ps, st))[1], k)[1]




first(Zygote.jacobian(x -> first(v_f_nn(x, vf_ps, vf_st)), [10.0]))


v_f_deriv([1.0], vf_ps, vf_st)

v_f(k, ps, st) = Lux.apply(v_f_nn, k, ps, st)

v_f([1.0], vf_ps, vf_st)

v_f_deriv([4.0], vf_ps, vf_st)
pol_f([1.0], pol_ps, pol_st)

Lux.apply(pol_f_nn, [1.0], pol_ps, pol_st)[1][1]


err_HJB([10.0], m, v_f([10.0], vf_ps, vf_st)[1], v_f_deriv([10.0], vf_ps, vf_st)[1], pol_f([10.0], pol_ps, pol_st)[1])

param_vec = ComponentArray(vf = vf_ps, pol = pol_ps)
states = (vf_st, pol_st)
opt = Optimisers.ADAM()

st_opt = Optimisers.setup(ADAM(), param_vec)


v_f_deriv([10.0], vf_ps, vf_st)

function loss_fn(x, m, vf_ps, pol_ps, vf_st, pol_st)
    v_f_k, vf_st_ = Lux.apply(v_f_nn, x, vf_ps, vf_st)
    pol_f_k, pol_st_ = Lux.apply(pol_f_nn, x, pol_ps, pol_st)
    # v_f_deriv_k = first(Zygote.jacobian(x -> first(v_f_nn(x, vf_ps, vf_st)), x))
    v_f_deriv_k = v_f_deriv(x, vf_ps, vf_st)
    hjb_err, pol_err = err_HJB(x, m, v_f_k, v_f_deriv_k, pol_f_k)

    loss = sum(hjb_err .^ 2) + sum(pol_err .^ 2)
    return loss, vf_st_, pol_st_
end

loss_fn([0.2], m, vf_ps, pol_ps, vf_st, pol_st)

epoch = 1
for epoch in 1:1000
    (loss, states...), back = Zygote.pullback(param_vec) do p
        loss_fn([10.0], m, p.vf, p.pol, states[1], states[2])
    end
    grads = back((1.0, nothing, nothing))[1]
    # epoch % 100 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")
    println("Epoch: $(epoch) | Loss: $(loss)")
    Optimisers.update!(st_opt, param_vec, grads)
end




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
