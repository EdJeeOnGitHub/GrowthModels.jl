
#### Utils for instantiating the model -----------------------------------------
function StochasticNPAbilityModel(
    stochasticprocess::Union{StochasticProcess,Nothing}; 
     γ = 2.0, ρ = 0.05, δ = 0.05, f = (x, z, η) -> (x .^0.1) .* z .* η, η_mean = 1.0)
    if isnothing(stochasticprocess)
        stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ =  0.1)
    end
    StochasticNPAbilityModel(γ, ρ, δ, f, η_mean, stochasticprocess)
end

function StochasticNPAbilityModel(;γ = 2.0, ρ = 0.05, δ = 0.05, f = (x, z, η) -> (x .^ 0.1) .* z .* η , η_mean = 1.0, θ = -log(0.9), σ = 0.1)
    StochasticNPAbilityModel(γ, ρ, δ, f, η_mean, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end

# let us pass param vecs
function StochasticNPAbilityModel(γ, ρ, δ, f, η_mean, θ, σ)
    StochasticNPAbilityModel(γ, ρ, δ, f, η_mean, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end


#### Steady State + Star Helpers -----------------------------------------------
# OrnsteinUhlenbeckProcess
k_steady_state_hi(m::StochasticNPAbilityModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess}= error("Not implemented")
k_steady_state_lo(m::StochasticNPAbilityModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = error("Not implemented")
# PoissonProcess
k_steady_state_hi(m::StochasticNPAbilityModel{T, S}) where {T <: Real, S <: PoissonProcess} = error("Not implemented")
k_steady_state_lo(m::StochasticNPAbilityModel{T, S}) where {T <: Real, S <: PoissonProcess} = error("Not implemented")


k_steady_state(m::StochasticNPAbilityModel) = [k_steady_state_lo(m), k_steady_state_hi(m)]
k_star(m::StochasticNPAbilityModel) = error("Not implemented")





#### Production Function -------------------------------------------------------
production_function(m::StochasticNPAbilityModel{T, S}, k, z, η) where {T <: Real, S <: StochasticProcess} = m.f(k, z, η)
