
#### Util Functions to Instantiate Model ---------------------------------------
function StochasticRamseyCassKoopmansModel(
    stochasticprocess::Union{StochasticProcess,Nothing}; 
     γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6)
    if isnothing(stochasticprocess)
        stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ =  0.1)
    end
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, stochasticprocess)
end


function StochasticRamseyCassKoopmansModel(
    γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6, θ = -log(0.9), σ = 0.1)
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end

#### K Steady State + Star Helpers ---------------------------------------------
# OrnsteinUhlenbeckProcess
k_steady_state_hi(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) 
k_steady_state_lo(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) 
# Poisson
k_steady_state_hi(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.α*m.A/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)
k_steady_state_lo(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.α*m.A/(m.ρ + m.δ))^(1/(1-m.α))  + process_mean(m.stochasticprocess)

k_star(m::StochasticRamseyCassKoopmansModel{T, S})  where {T <: Real, S <: OrnsteinUhlenbeckProcess}= (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))
k_star(m::StochasticRamseyCassKoopmansModel{T, S})  where {T <: Real, S <: PoissonProcess} = (m.α*m.A/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)

k_steady_state(m::StochasticRamseyCassKoopmansModel) = vcat(k_steady_state_lo(m), k_steady_state_hi(m))


#### Production Function -------------------------------------------------------
# Production function
@inline function stochastic_rck_production_function(::Union{OrnsteinUhlenbeckProcess,Type{OrnsteinUhlenbeckProcess}}, k, z, α, A)
     A .* z .* k .^ α
end
# Poisson
@inline function stochastic_rck_production_function(::Union{PoissonProcess,Type{PoissonProcess}}, k, z, α, A)
     A .* k .^ α .+ z
end
# Derivative of production function
@inline function stochastic_rck_production_function_prime(::Union{OrnsteinUhlenbeckProcess,Type{OrnsteinUhlenbeckProcess}}, k, z, α, A)
     A .* z .* α .* k .^ (α - 1)
end
# Poisson
@inline function stochastic_rck_production_function_prime(::Union{PoissonProcess,Type{PoissonProcess}}, k, z, α, A)
     A .* α .* k .^ (α - 1)
end

# Method dispatch stuff
@inline production_function(::StochasticRamseyCassKoopmansModel{T, S}, k, z, α::Real, A::Real) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function(S, k, z, α, A)
@inline production_function(::StochasticRamseyCassKoopmansModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function(S, k, z, params[1], params[2])
@inline production_function(m::StochasticRamseyCassKoopmansModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function(m.stochasticprocess, k, z, m.α, m.A)
@inline production_function(::StochasticRamseyCassKoopmansModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = production_function(S, x[:, 1], x[:, 2], params[1], params[2])

@inline production_function_prime(::StochasticRamseyCassKoopmansModel{T, S}, k, z, α::Real, A::Real, δ::Real) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function_prime(S, k, z, α, A)
@inline production_function_prime(::StochasticRamseyCassKoopmansModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function_prime(S, k, z, params[1], params[2])
@inline production_function_prime(m::StochasticRamseyCassKoopmansModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function_prime(m.stochasticprocess, k, z, m.α, m.A)
@inline production_function_prime(::StochasticRamseyCassKoopmansModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = production_function_prime(S, x[:, 1], x[:, 2], params[1], params[2])



function plot_production_function(m::StochasticRamseyCassKoopmansModel, k, z)
    y = production_function(m, collect(k), collect(z)')
    plot(k, y, label="")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end