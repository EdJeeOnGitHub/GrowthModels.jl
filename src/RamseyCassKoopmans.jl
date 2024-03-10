# NGM model




# HyperParams for the NGM model
function HyperParams(m::RamseyCassKoopmansModel; N = 1000, kmax_f = 1.3, kmin_f = 0.001)
    kss = k_steady_state(m)
    kmin, kmax = kmin_f*kss, kmax_f*kss
    dk = (kmax-kmin)/(N-1)
    HyperParams(N, dk, kmax, kmin)
end


function StateSpace(m::RamseyCassKoopmansModel, hyperparams::HyperParams)
    k = range(hyperparams.kmin, hyperparams.kmax, length = hyperparams.N)
    y = production_function(m, collect(k))
    StateSpace((k = k, y = y))
end


function RamseyCassKoopmansModel(; γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6)
    RamseyCassKoopmansModel(γ, α, ρ, δ, A)
end

k_steady_state_RCK(α::Real, A::Real, ρ::Real, δ::Real) = (α*A/(ρ + δ))^(1/(1-α))
k_steady_state(m::RamseyCassKoopmansModel) = k_steady_state_RCK(m.α, m.A, m.ρ, m.δ) 
k_steady_state_hi_RCK(α::Real, A::Real, ρ::Real, δ::Real) = k_steady_state_RCK(α, A, ρ, δ)
k_steady_state_hi(m::RamseyCassKoopmansModel) = k_steady_state(m)
k_steady_state_lo(m::RamseyCassKoopmansModel) = k_steady_state(m)
k_steady_state_lo_RCK(α::Real, A::Real, ρ::Real, δ::Real) = k_steady_state_RCK(α, A, ρ, δ)
k_star(m::RamseyCassKoopmansModel) = k_steady_state(m)
k_star_RCK(α::Real, A::Real, ρ::Real, δ::Real) = k_steady_state_RCK(α, A, ρ, δ)

# Production function
@inline function rck_production_function(k, α, A, δ)
    A * pow(k, α)
end
# Derivative of production function
@inline function rck_production_function_prime(k, α, A, δ)
    A * α * pow(k, α - 1)
end

@inline production_function(::RamseyCassKoopmansModel, k::Union{Real,Vector{<:Real}}, α::Real, A::Real, δ::Real) = rck_production_function.(k, α, A, δ)
@inline production_function(::RamseyCassKoopmansModel, k::Union{Real,Vector{<:Real}}, params::Vector) = rck_production_function.(k, params[1], params[2], params[3])
@inline production_function(m::RamseyCassKoopmansModel, k::Union{Real,Vector{<:Real}}) = rck_production_function.(k, m.α, m.A, m.δ)

@inline production_function_prime(::RamseyCassKoopmansModel, k::Union{Real,Vector{<:Real}}, α::Real, A::Real, δ::Real) = rck_production_function_prime.(k, α, A, δ)
@inline production_function_prime(::RamseyCassKoopmansModel, k::Union{Real,Vector{<:Real}}, params::Vector) = rck_production_function_prime.(k, params[1], params[2], params[3])
@inline production_function_prime(m::RamseyCassKoopmansModel, k::Union{Real,Vector{<:Real}}) = rck_production_function_prime.(k, m.α, m.A, m.δ)




#### Misc Functions ####
k_dot(m::RamseyCassKoopmansModel) =  (variables::NamedTuple) -> variables.y .- m.δ .* variables.k .- variables.c

#### Plotting ####
function plot_production_function(m::RamseyCassKoopmansModel, k)
    y = production_function(m, collect(k))
    plot(k, y, label="y")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end