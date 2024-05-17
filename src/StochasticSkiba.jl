
#### Utils for instantiating the model -----------------------------------------
function StochasticSkibaModel(
    stochasticprocess::Union{StochasticProcess,Nothing}; 
     γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0)
    if isnothing(stochasticprocess)
        stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ =  0.1)
    end
    StochasticSkibaModel(γ, α, ρ, δ, A_H, A_L, κ, stochasticprocess)
end

function StochasticSkibaModel(;γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0, θ = -log(0.9), σ = 0.1)
    StochasticSkibaModel(γ, α, ρ, δ, A_H, A_L, κ, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end

# let us pass param vecs
function StochasticSkibaModel(γ, α, ρ, δ, A_H, A_L, κ, θ, σ)
    StochasticSkibaModel(γ, α, ρ, δ, A_H, A_L, κ, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end


#### Steady State + Star Helpers -----------------------------------------------

# OrnsteinUhlenbeckProcess
k_steady_state_hi(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess}= (m.α*m.A_H*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.α*m.A_L*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))
# PoissonProcess
k_steady_state_hi(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: PoissonProcess}= (m.α*m.A_H/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ + process_mean(m.stochasticprocess)
k_steady_state_lo(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.α*m.A_L/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)


k_steady_state(m::StochasticSkibaModel) = [k_steady_state_lo(m), k_steady_state_hi(m)]
k_star(m::StochasticSkibaModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))

#### Production Function -------------------------------------------------------
# OU uses productivity shock
@inline function stochastic_skiba_production_function(::Union{OrnsteinUhlenbeckProcess,Type{OrnsteinUhlenbeckProcess}}, k, z, α, A_H, A_L, κ)
    z .* max.(A_H .* max.(k .- κ, 0) .^ α, A_L .* k .^ α)
end
# PoissonProcess just an income shock
@inline function stochastic_skiba_production_function(::Union{PoissonProcess,Type{PoissonProcess}}, k, z, α, A_H, A_L, κ)
     max.(max.(A_H .* max.(k .- κ, 0) .^ α, A_L .* k .^ α) .+ z, 1e-3)
end
# derivative of skiba production function
@inline function stochastic_skiba_production_function_prime(::Union{OrnsteinUhlenbeckProcess,Type{OrnsteinUhlenbeckProcess}}, k, z, α, A_H, A_L, κ)
    if k .> κ
        z .* A_H .* α .* (k .- κ) .^ (α - 1)
    else
        z .* A_L .* α .* k .^ (α - 1)
    end
end

@inline function stochastic_skiba_production_function_prime(::Union{PoissonProcess,Type{PoissonProcess}}, k, z, α, A_H, A_L, κ)
    if k .> κ
         A_H .* α .* (k .- κ) .^ (α - 1)
    else
         A_L .* α .* k .^ (α - 1)
    end
end
# Method dispatch for various forms of arguments
# k, z, and params passed directly
@inline production_function(::StochasticSkibaModel{T,S}, k, z, α::Real, A_H::Real, A_L::Real, κ::Real) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(S, k, z, α, A_H, A_L, κ)
@inline production_function_prime(::StochasticSkibaModel{T, S}, k, z, α::Real, A_H::Real, A_L::Real, κ::Real) where {T <: Real, S <: StochasticProcess} =  stochastic_skiba_production_function_prime(S, k, z, α, A_H, A_L, κ)
# k, z, and params passed as a vector
@inline production_function(::StochasticSkibaModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(S, k, z, params[1], params[2], params[3], params[4])
@inline production_function_prime(::StochasticSkibaModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function_prime(S, k, z, params[1], params[2], params[3], params[4])
# k, z, and params passed using model fields
@inline production_function(m::StochasticSkibaModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(m.stochasticprocess, k, z, m.α, m.A_H, m.A_L, m.κ)
@inline production_function_prime(m::StochasticSkibaModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function_prime(m.stochasticprocess, k, z, m.α, m.A_H, m.A_L, m.κ)
# k, z passed as matrix and params using vector
@inline production_function(::StochasticSkibaModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(S, x[:,1], x[:,2], params[1], params[2], params[3], params[4])
@inline production_function_prime(::StochasticSkibaModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function_prime(S, x[:,1], x[:,2], params[1], params[2], params[3], params[4])





