# This is a fixed point iteration loop to solve the Hamilton Jacobi BellmanPDE
# for the Neoclassical Growth Model with a convex-concave production function
# as in Skiba (1978) "Optimal Growth with a Convex-Concave Production Function"
# Use "butterfly technology":
# f(k) = max(f_H(k),f_L(k)), f_H(k) = A_H*max(k - kappa,0).^a, f_L(k) = A_L*k^a;
# Matlab code originally written by Greg Kaplan and Benjamin Moll - thanks fellas



#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function HyperParams(m::SkibaModel; N = 1000, kmax_f = 1.3, kmin_f = 0.001)
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    dk = (kmax-kmin)/(N-1)
    HyperParams(N, dk, kmax, kmin)
end

function StateSpace(m::SkibaModel, hyperparams::HyperParams)
    k = range(hyperparams.kmin, hyperparams.kmax, length = hyperparams.N)
    y = production_function(m, collect(k))
    StateSpace((k = k, y = y))
end



function SkibaModel(; γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0)
    SkibaModel(γ, α, ρ, δ, A_H, A_L, κ)
end




k_steady_state_hi(m::SkibaModel) = (m.α*m.A_H/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::SkibaModel) = (m.α*m.A_L/(m.ρ + m.δ))^(1/(1-m.α))
k_star(m::SkibaModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))


y_H(m::SkibaModel) = (k) -> m.A_H*max(k - m.κ,0)^m.α
y_L(m::SkibaModel) = (k) -> m.A_L*k^m.α 


@inline function skiba_production_function(k, α, A_H, A_L, κ)
    max(A_H * pow(max(k - κ, 0), α), A_L * pow(k, α))
end

@inline production_function(m::SkibaModel, k::Real, α::Real, A_H::Real, A_L::Real, κ::Real) = skiba_production_function(k, α, A_H, A_L, κ)
@inline production_function(m::SkibaModel, k::Real, params::Vector) = skiba_production_function(k, params[1], params[2], params[3], params[4])
@inline production_function(m::SkibaModel, k::Real) = skiba_production_function(k, m.α, m.A_H, m.A_L, m.κ)

@inline production_function(m::SkibaModel, k::Vector, α::Real, A_H::Real, A_L::Real, κ::Real) = skiba_production_function.(k, α, A_H, A_L, κ)
@inline production_function(m::SkibaModel, k::Vector, params::Vector) = skiba_production_function.(k, params[1], params[2], params[3], params[4])
@inline production_function(m::SkibaModel, k::Vector) = skiba_production_function.(k, m.α, m.A_H, m.A_L, m.κ)










#### Misc Functions ####
k_dot(m::SkibaModel) = (variables::NamedTuple) -> variables.y .- m.δ .* variables.k .- variables.c

#### Plotting ####
function plot_production_function(m::SkibaModel, k)
    y = production_function(m, collect(k))
    yH = y_H(m).(k)
    yL = y_L(m).(k)
    plot(k, y, label="y")
    plot!(k, yH, linestyle=:dash, label="yH")
    plot!(k, yL, linestyle=:dash, label="yL")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end

