using ComponentArrays, Lux, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots
using GrowthModels
using Zygote
using ForwardDiff, LinearAlgebra
using LuxCUDA
using Plots


Random.seed!(1234)
device = cpu_device()
# Stuff for GPU
skiba_production_function(k, α, A_H, A_L, κ) = max(A_H .* max(k - κ, 0).^α, A_L .* (k .^ α))
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
    # exp to ensure positive in input layer
    y = (abs.(ps.weight) * x) .+ ps.bias
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

m = SkibaModel()
isa(m, SkibaModel)
params(m)


function (l::MicawberLayer)(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple, m::Model) where {T <: Real}
    k = x[1, :]
    γ, α, ρ, δ, A_H, A_L, κ = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :], x[7, :], x[8, :]
    k_s = κ ./ (1 .- (A_L ./ A_H).^(1 ./ α))

    # abs to ensure positive in input layer
    # difference between x and k_star
    y = (abs.(ps.weight) * (k .- k_s)) .+ ps.bias
    return l.activation.(y), st
end

function (l::SteadyStateLayer)(x::Union{AbstractVecOrMat, T}, ps, st::NamedTuple, m::Model) where {T <: Real}
    k = x[1, :]
    γ, α, ρ, δ, A_H, A_L, κ = x[2, :], x[3, :], x[4, :], x[5, :], x[6, :], x[7, :], x[8, :]
    # abs to ensure positive in input layer
    # difference between x and k_star
    k_ss_hi = k_steady_state_hi_Skiba.(α, A_H, ρ, δ, κ)
    k_ss_lo = k_steady_state_lo_Skiba.(α, A_L, ρ, δ)
    k_ss = [k_ss_lo, k_ss_hi] |> device
    diff = -1.0f32 .* (k_ss .- k)
    abs_diff = abs.(diff)
    distances_indices = argmin(abs_diff, dims = 1)
    distances = diff[distances_indices]

    y = (abs.(ps.weight) * distances) .+ ps.bias
    return l.activation.(y), st
end


function err_HJB(k, model, v_f_k, v_f_deriv_k, pol_f_k)
    (; ρ, δ, γ) = model
    (; γ, α, ρ, δ, A_H, A_L, κ) = model

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
model_params = params(m)
n_params = length(model_params)

batch_size = 1_00

grid_vals = reshape(collect(range(0.1, 2*k_star(m), length = batch_size)), (1, batch_size))
grid_vals = sort(grid_vals, dims =1)
param_grid = repeat(model_params, 1, batch_size)
vals = vcat(grid_vals, param_grid)
vals = abs.(vals)
cpu_vals = vals
vals = vals |> device
vec_vals = vec(vals[1, :])
c_size = 24
n_size = 48

v_f_nn = Chain(
    # both read in simultaneously
    Parallel(
        nothing,
        MicawberLayer(1 + n_params, c_size, tanh, m),
        SteadyStateLayer(1 + n_params, c_size, tanh, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    PositiveDense(c_size*2 + 1, n_size),
    PositiveDense(n_size, n_size),
    PositiveDense(n_size, 1)
)

pol_f_nn = Chain(
    Parallel(
        nothing,
        SteadyStateLayer(1 + n_params, c_size, tanh, m),
        MicawberLayer(1 + n_params, c_size, tanh, m),
        NoOpLayer()
    ),
    x -> vcat(x...),
    Dense(c_size*2 + 1 => n_size, tanh),
    Dense(n_size => n_size, tanh),
    Dense(n_size => 1, softplus),
)

rng = Random.default_rng()
cpu_vf_ps, cpu_vf_st = Lux.setup(rng, v_f_nn) 
cpu_pol_ps, cpu_pol_st = Lux.setup(rng, pol_f_nn) 

vf_ps, vf_st = Lux.setup(rng, v_f_nn) .|> device
pol_ps, pol_st = Lux.setup(rng, pol_f_nn) .|> device

vf_y, vf_st = Lux.apply(v_f_nn, vals, vf_ps, vf_st)


function v_f(k, ps, st)
    return first(Lux.apply(v_f_nn, k, ps, st)) 
end

function pol_f(k, ps, st)
    return first(Lux.apply(pol_f_nn, k, ps, st)) 
end
function v_f_scalar(k, ps, st)
    first(v_f(k, ps, st))
end

# function v_f_deriv_scalar(k, ps, st)
#     v_ed(x) = v_f_scalar(x, ps, st)
#     # d = Zygote.forwarddiff(k) do k_
#     #     ForwardDiff.derivative(v_ed, k_)
#     # end
#     d = ForwardDiff.derivative(v_ed, k)
#     return d
# end
# function v_f_deriv(k, ps, st)
#     v_f_deriv_scalar.(k, Ref(ps), Ref(st))
# end

v_f_deriv(k, ps, st) = [Zygote.forwarddiff(z -> ForwardDiff.derivative(x -> first(v_f(x, ps, st)), z), y) for y in k] |> device
# v_f_deriv(k, ps, st) = [Zygote.forwarddiff(z -> ForwardDiff.derivative(x -> v_f([x], ps, st)[1], z), y) for y in k]
using BenchmarkTools
# v_f_deriv_scalar(1.0, vf_ps, vf_st)
# @btime v_f_deriv(vals, vf_ps, vf_st);
# @btime old_v_f_deriv(vals, vf_ps, vf_st);

# @btime v_f_deriv(vals, vf_ps, vf_st);

l, b = Zygote.pullback(vf_ps) do p
    v_f_deriv(vals, p, vf_st)
end;


using Enzyme

# l
# b(vf_ps)


# @btime l, b = Zygote.pullback(vf_ps) do p
#     old_v_f_deriv(vals, p, vf_st)
# end;
# l
# b(vf_ps)

# v_f_deriv(k, ps, st) = [Zygote.forwarddiff(z -> ForwardDiff.derivative(x -> v_f([x], ps, st)[1], z), y) for y in k]

# function dfx(f, k, p, st)
#     _, back = Zygote.pullback(k) do k_
#         Zygote.forwarddiff(k_) do k_
#             f(k_, p, st)
#         end
#     end
#     return back(ones(size(k)))[1]
# end
# l, b = Zygote.pullback(vf_ps) do p
#     dfx(v_f, vals, p, vf_st)
# end;
# b(ones(size(vals)))
# function reverse_dfx(f, k, p, st)
#     _, back = Zygote.pullback(k) do k_
#             f(k_, p, st)
#     end
#     return back(ones(size(k)))[1] 
# end

# dfx(v_f, vals, vf_ps, vf_st)
# using BenchmarkTools
# @btime v_f(vals, vf_ps, vf_st);
# @btime dfx(v_f, vals, vf_ps, vf_st);
# @btime reverse_dfx(v_f, vals, vf_ps, vf_st);

# v_f(vals, vf_ps, vf_st)
# dfx(v_f, vals, vf_ps, vf_st)
# reverse_dfx(v_f, vals, vf_ps, vf_st)

# l, b = Zygote.pullback(vf_ps) do p
#     dfx(v_f, vals, p, vf_st)
# end;

# l
# b(vals)

# @btime l, b = Zygote.pullback(vf_ps) do p
#     dfx(v_f, vals, p, vf_st)
# end;

# @btime rev_l, rev_b = Zygote.pullback(vf_ps) do p
#     reverse_dfx(v_f, vals, p, vf_st)
# end;

# b(ones(size(vals)))
# rev_b(ones(size(vals)))

# b(1.0)

# using BenchmarkTools
# v_f(vals, vf_ps, vf_st)
# pol_f(vals, pol_ps, pol_st)
# @btime v_f_deriv(vals, vf_ps, vf_st);



using BenchmarkTools
# @btime v_f(vals, vf_ps, vf_st);
# @btime pol_f(vals, pol_ps, pol_st);
# @btime v_f_deriv(vals, vf_ps, vf_st);


# param_vec = ComponentArray(vf = vf_ps, pol = pol_ps) |> device
states = (vf_st, pol_st) |> device
params = (vf_ps, pol_ps) |> device
opt = Optimisers.ADAM() 

st_opt = Optimisers.setup(ADAM(), params) |> device



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

    # kdot = production_function(m, vec_x) .- m.δ .* vec_x .- pol_f_k
    # kt1 = vec_x .+ kdot
    n_k = length(vec_x)
    loss = sqrt(sum(abs2, hjb_err) / n_k)  + sqrt(sum(abs2, pol_err)  / n_k)
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
v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, vf_ps, pol_ps, vf_st, pol_st)



function plot_pred_output(vals, v_f_k, v_f_deriv_k, pol_f_k)
    p1 = plot(
        vec(vals),
        v_f_k,
        seriestype = :scatter,
        colour = :blue,
        label = "NN \$V(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$V(k)\$",
        )
    p2 = plot(
        vec(vals),
        v_f_deriv_k,
        seriestype = :scatter,
        colour = :blue,
        label = "NN \$V'(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$V'(k)\$",
        )
    p3 = plot(
        vec(vals),
        pol_f_k,
        seriestype = :scatter,
        colour = :blue,
        label = "NN \$c(k)\$",
        xlabel = "\$k\$",
        ylabel = "\$c(k)\$",
        )
    return p1, p2, p3
end

plot_pred_output(vals, v_f_k, v_f_deriv_k, pol_f_k)
loss_fn(vals, m, vf_ps, pol_ps, vf_st, pol_st)


l, back = Zygote.pullback(params) do p
    loss_fn(vals, m, p[1], p[2], states[1], states[2])
end;

l
back(1.0)

epoch_list = [0]
loss_list = [Inf]
for epoch in epoch_list[end]:1_000_000
# epoch = 1
    loss, back = Zygote.pullback(params) do p
        loss_fn(vals, m, p[1], p[2], states[1], states[2])
    end
    grads = back(1.0)[1]
    epoch % 500 == 1 && println("Epoch: $(epoch) | Loss: $(loss)")

    if epoch % 500 == 1
        push!(epoch_list, epoch)
        push!(loss_list, loss)
        v_f_k, v_f_deriv_k, pol_f_k = predict_fn(vals, params[1], params[2], states[1], states[2])
        kdot = production_function(m, vec_vals) .- m.δ .* vec_vals .- pol_f_k
        hjb_err, pol_err = err_HJB(vec(vals), m, v_f_k, v_f_deriv_k, pol_f_k)


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
        display(p_all)
    end
    # println("Epoch: $(epoch) | Loss: $(loss)")
    Optimisers.update!(st_opt, params, grads)
end

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