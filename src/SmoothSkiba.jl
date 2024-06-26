# This is a fixed point iteration loop to solve the Hamilton Jacobi BellmanPDE
# for the Neoclassical Growth Model with a convex-concave production function
# as in Skiba (1978) "Optimal Growth with a Convex-Concave Production Function"
# Using "softmax butterfly technology":
# f(k) = (ω(k)A_H + (1 - ω(k))A_L)k^α; where ω(k) = 1 / (1 + exp(-β(k - κ)))
# β → -∞ gives a weird technology where productivity is initially high, and then falls
# β -> ∞ gives the original Skiba butterfly technology with a max-like operator
# β = 0 evenly weights the two technologies with no level effect (CB)
# β > 0 gives a max-like butterfly production function but with smooth sigmoid 
# transition between the two technologies. The higher $\beta$ the steeper the transition.
# Matlab code originally written by Greg Kaplan and Benjamin Moll - thanks fellas


#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function StateSpaceHyperParams(m::SmoothSkibaModel; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001)
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin)
    return StateSpaceHyperParams((k = k_hps,))
end

function StateSpace(m::SmoothSkibaModel, statespacehyperparams::StateSpaceHyperParams)
    k_hps = statespacehyperparams[:k]
    k = collect(range(k_hps.xmin, k_hps.xmax, length = k_hps.N))
    y = production_function(m, k)
    StateSpace((k = k,), (y = y,))
end



function SmoothSkibaModel(; γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0, β = 10.0)
    SmoothSkibaModel(γ, α, ρ, δ, A_H, A_L, κ, β)
end


y_H(m::SmoothSkibaModel) = (k) -> m.A_H*k^m.α
y_L(m::SmoothSkibaModel) = (k) -> m.A_L*k^m.α 
# Just an approximation -> assuming ω(k) == 1
k_steady_state_hi_SmoothSkiba(α::Real, A_H::Real, ρ::Real, δ::Real, κ::Real) = (α*A_H/(ρ + δ))^(1/(1-α)) + κ
k_steady_state_hi(m::SmoothSkibaModel) = k_steady_state_hi_SmoothSkiba(m.α, m.A_H, m.ρ, m.δ, m.κ) 
# Just an approximation -> assuming ω(k) == 0
k_steady_state_lo_SmoothSkiba(α::Real, A_L::Real, ρ::Real, δ::Real, κ::Real) = (α*A_L/(ρ + δ))^(1/(1-α))
k_steady_state_lo(m::SmoothSkibaModel) = k_steady_state_lo_SmoothSkiba(m.α, m.A_L, m.ρ, m.δ, m.κ) 
# not really well defined atm -> think about this ed
k_star(m::SmoothSkibaModel) = k_star_SmoothSkiba(m.α, m.A_L, m.A_H, m.κ)
k_star_SmoothSkiba(α::Real, A_L::Real, A_H::Real, κ::Real) = κ/(1-(A_L/A_H)^(1/α))


k_steady_state(m::SmoothSkibaModel) = [k_steady_state_lo(m), k_steady_state_hi(m)]

# Skiba production function
@inline function smooth_skiba_production_function(k, α, A_H, A_L, κ, β)
    weight_fun = 1 / (1 + exp(-β * (k - κ)))
    A = A_H * weight_fun + A_L * (1 - weight_fun)
    return A * k^α
end
# derivative of skiba production function
@inline function smooth_skiba_production_function_prime(k, α, A_H, A_L, κ, β)
    weight_fun = 1 / (1 + exp(-β * (k - κ)))
    exp_deriv = β * exp(-β * (k - κ)) / (1 + exp(-β * (k - κ)))^2
    y_prime = (A_H - A_L)*exp_deriv * k^α + (A_H * weight_fun + A_L * (1 - weight_fun)) * α * k^(α - 1)
    return y_prime
end


@inline production_function(::SmoothSkibaModel, k::Union{Real,Vector{<:Real}}, α::Real, A_H::Real, A_L::Real, κ::Real, β::Real) = smooth_skiba_production_function.(k, α, A_H, A_L, κ, β);
@inline production_function(::SmoothSkibaModel, k::Union{Real,Vector{<:Real}}, params::Vector) = smooth_skiba_production_function.(k, params[1], params[2], params[3], params[4], params[5])
@inline production_function(m::SmoothSkibaModel, k::Union{Real,Vector{<:Real}}) = smooth_skiba_production_function.(k, m.α, m.A_H, m.A_L, m.κ, m.β)

@inline production_function_prime(::SmoothSkibaModel, k::Union{Real,Vector{<:Real}}, α::Real, A_H::Real, A_L::Real, κ::Real, β::Real) = smooth_skiba_production_function_prime.(k, α, A_H, A_L, κ, β)
@inline production_function_prime(::SmoothSkibaModel, k::Union{Real,Vector{<:Real}}, params::Vector) = smooth_skiba_production_function_prime.(k, params[1], params[2], params[3], params[4], params[5])
@inline production_function_prime(m::SmoothSkibaModel, k::Union{Real,Vector{<:Real}}) = smooth_skiba_production_function_prime.(k, m.α, m.A_H, m.A_L, m.κ, m.β)



#### Plotting ####
k_dot(m::SmoothSkibaModel) = (variables::NamedTuple) -> variables.y .- m.δ .* variables.k .- variables.c
function plot_production_function(m::SmoothSkibaModel, k)
    y = production_function(m, collect(k))
    yH = y_H(m).(k)
    yL = y_L(m).(k)
    plot(k, y, label="y")
    plot!(k, yH, linestyle=:dash, label="yH")
    plot!(k, yL, linestyle=:dash, label="yL")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end

