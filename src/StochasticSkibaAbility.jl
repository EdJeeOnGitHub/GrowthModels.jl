
#### Utils for instantiating the model -----------------------------------------
function StochasticSkibaAbilityModel(
    stochasticprocess::Union{StochasticProcess,Nothing}; 
     γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0, η_mean = 1.0)
    if isnothing(stochasticprocess)
        stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ =  0.1)
    end
    StochasticSkibaAbilityModel(γ, α, ρ, δ, A_H, A_L, κ, η_mean, stochasticprocess)
end

function StochasticSkibaAbilityModel(;γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0, η_mean = 1.0, θ = -log(0.9), σ = 0.1)
    StochasticSkibaAbilityModel(γ, α, ρ, δ, A_H, A_L, κ, η_mean, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end

# let us pass param vecs
function StochasticSkibaAbilityModel(γ, α, ρ, δ, A_H, A_L, κ, η_mean, θ, σ)
    StochasticSkibaAbilityModel(γ, α, ρ, δ, A_H, A_L, κ, η_mean, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end


#### Steady State + Star Helpers -----------------------------------------------

# OrnsteinUhlenbeckProcess
k_steady_state_hi(m::StochasticSkibaAbilityModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess}= (m.η_mean*m.α*m.A_H*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::StochasticSkibaAbilityModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.η_mean*m.α*m.A_L*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))
# PoissonProcess
k_steady_state_hi(m::StochasticSkibaAbilityModel{T, S}) where {T <: Real, S <: PoissonProcess}= (m.η_mean*m.α*m.A_H/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ + process_mean(m.stochasticprocess)
k_steady_state_lo(m::StochasticSkibaAbilityModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.η_mean*m.α*m.A_L/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)


k_steady_state(m::StochasticSkibaAbilityModel) = [k_steady_state_lo(m), k_steady_state_hi(m)]
k_star(m::StochasticSkibaAbilityModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))

#### Production Function -------------------------------------------------------
# OU uses productivity shock
@inline function stochastic_skiba_ability_production_function(::Union{OrnsteinUhlenbeckProcess,Type{OrnsteinUhlenbeckProcess}}, k, z, η, α, A_H, A_L, κ)
    z .* η .* max.(A_H .* max.(k .- κ, 0) .^ α, A_L .* k .^ α)
end
# PoissonProcess just an income shock
@inline function stochastic_skiba_ability_production_function(::Union{PoissonProcess,Type{PoissonProcess}}, k, z, η, α, A_H, A_L, κ)
     max.(max.(η .* A_H .* max.(k .- κ, 0) .^ α,  η .* A_L .* k .^ α) .+ z, 1e-3)
end
# derivative of skiba production function
@inline function stochastic_skiba_ability_production_function_prime(::Union{OrnsteinUhlenbeckProcess,Type{OrnsteinUhlenbeckProcess}}, k, z, η, α, A_H, A_L, κ)
    if k .> κ
        η .* z .* A_H .* α .* (k .- κ) .^ (α - 1)
    else
        η .* z .* A_L .* α .* k .^ (α - 1)
    end
end

@inline function stochastic_skiba_ability_production_function_prime(::Union{PoissonProcess,Type{PoissonProcess}}, k, z, η, α, A_H, A_L, κ)
    if k .> κ
         η .* A_H .* α .* (k .- κ) .^ (α - 1)
    else
        η .* A_L .* α .* k .^ (α - 1)
    end
end

# Method dispatch for various forms of arguments
# k, z, and params passed directly
@inline production_function(::StochasticSkibaAbilityModel{T,S}, k, z, η, α::T, A_H::T, A_L::T, κ::T) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function(S, k, z, η, α, A_H, A_L, κ)
@inline production_function_prime(::StochasticSkibaAbilityModel{T, S}, k, z, η, α::T, A_H::T, A_L::T, κ::T) where {T <: Real, S <: StochasticProcess} =  stochastic_skiba_ability_production_function_prime(S, k, z, η, α, A_H, A_L, κ)
# k, z, and params passed as a vector
@inline production_function(::StochasticSkibaAbilityModel{T, S}, k, z, η, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function(S, k, z, η, params[1], params[2], params[3], params[4])
@inline production_function_prime(::StochasticSkibaAbilityModel{T, S}, k, z, η, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function_prime(S, k, z, η, params[1], params[2], params[3], params[4])
# k, z, and params passed using model fields
@inline production_function(m::StochasticSkibaAbilityModel{T, S}, k, z, η) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function(m.stochasticprocess, k, z, η, m.α, m.A_H, m.A_L, m.κ)
@inline production_function_prime(m::StochasticSkibaAbilityModel{T, S}, k, z, η) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function_prime(m.stochasticprocess, k, z, η, m.α, m.A_H, m.A_L, m.κ)
# k, z passed as matrix and params using vector
@inline production_function(::StochasticSkibaAbilityModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function(S, x[:,1], x[:,2], x[:, 3], params[1], params[2], params[3], params[4])
@inline production_function_prime(::StochasticSkibaAbilityModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_ability_production_function_prime(S, x[:,1], x[:,2], x[:, 3], params[1], params[2], params[3], params[4])