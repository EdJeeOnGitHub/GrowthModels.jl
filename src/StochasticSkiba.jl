

#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function StateSpaceHyperParams(m::StochasticSkibaModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin)
    # z_hps
    zmean = process_mean(m.stochasticprocess)
    zmin = zmean*0.8
    zmax = zmean*1.2
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps))
end

function StateSpaceHyperParams(m::StochasticSkibaModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 2) where {T <: Real, S <: PoissonProcess}
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin)
    # z_hps
    zmin = minimum(m.stochasticprocess.z)
    zmax = maximum(m.stochasticprocess.z)
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps))
end

function StateSpace(m::StochasticSkibaModel{T, S}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    k = collect(range(k_hps.xmin, k_hps.xmax, length = k_hps.N))
    z = collect(range(z_hps.xmin, z_hps.xmax, length = z_hps.N))
    # z' creates Nk x Nz matrix
    y = production_function(m, k, z')
    StateSpace((k = k, z = z), (y = y,))
end

function StateSpace(m::StochasticSkibaModel{T, S}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: PoissonProcess}
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    k = collect(range(k_hps.xmin, k_hps.xmax, length = k_hps.N))
    z = collect(range(z_hps.xmin, z_hps.xmax, length = z_hps.N))
    z = vcat(z_hps.xmin, z_hps.xmax)
    # z' creates Nk x Nz matrix
    y = production_function(m, k, z')
    StateSpace((k = k, z = z), (y = y,))
end

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




# OrnsteinUhlenbeckProcess
k_steady_state_hi(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess}= (m.α*m.A_H*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.α*m.A_L*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))
# PoissonProcess
k_steady_state_hi(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: PoissonProcess}= (m.α*m.A_H/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ + process_mean(m.stochasticprocess)
k_steady_state_lo(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.α*m.A_L/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)



k_steady_state(m::StochasticSkibaModel) = [k_steady_state_lo(m), k_steady_state_hi(m)]
k_star(m::StochasticSkibaModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))

y_H(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (k, z) -> m.A_H*z*max(k - m.κ,0)^m.α
y_L(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (k, z) -> m.A_L*z*k^m.α 

y_H(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: PoissonProcess} = (k, z) -> m.A_H*max(k - m.κ,0)^m.α + z
y_L(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: PoissonProcess} = (k, z) -> m.A_L*k^m.α  + z

# Skiba production function
@inline function stochastic_skiba_production_function(::OrnsteinUhlenbeckProcess, k, z, α, A_H, A_L, κ)
    z .* max.(A_H .* max.(k .- κ, 0) .^ α, A_L .* k .^ α)
end

@inline function stochastic_skiba_production_function(::PoissonProcess, k, z, α, A_H, A_L, κ)
     max.(max.(A_H .* max.(k .- κ, 0) .^ α, A_L .* k .^ α) .+ z, 1e-3)
end
# derivative of skiba production function
@inline function skiba_production_function_prime(::OrnsteinUhlenbeckProcess, k, z, α, A_H, A_L, κ)
    if k .> κ
        z .* A_H .* α .* pow.(k .- κ, α - 1)
    else
        z .* A_L .* α .* pow.(k, α - 1)
    end
end

@inline function skiba_production_function_prime(::Type{PoissonProcess}, k, z, α, A_H, A_L, κ)
    if k .> κ
         A_H .* α .* pow.(k .- κ, α - 1)
    else
         A_L .* α .* pow.(k, α - 1)
    end
end

@inline production_function(::StochasticSkibaModel{T,S}, k, z, α::Real, A_H::Real, A_L::Real, κ::Real) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(S(), k, z, α, A_H, A_L, κ)
@inline production_function(::StochasticSkibaModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(S(), k, z, params[1], params[2], params[3], params[4])
@inline production_function(m::StochasticSkibaModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_production_function(m.stochasticprocess, k, z, m.α, m.A_H, m.A_L, m.κ)

@inline production_function_prime(::StochasticSkibaModel{T, S}, k, z, α::Real, A_H::Real, A_L::Real, κ::Real) where {T <: Real, S <: StochasticProcess} =  skiba_production_function_prime(S(), k, z, α, A_H, A_L, κ)
@inline production_function_prime(::StochasticSkibaModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = skiba_production_function_prime(S(), k, z, params[1], params[2], params[3], params[4])
@inline production_function_prime(m::StochasticSkibaModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = skiba_production_function_prime(m.stochasticprocess, k, z, m.α, m.A_H, m.A_L, m.κ)





