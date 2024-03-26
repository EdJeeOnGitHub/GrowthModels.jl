

#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function StateSpaceHyperParams(m::StochasticSkibaModel; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40)
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

function StateSpace(m::StochasticSkibaModel, statespacehyperparams::StateSpaceHyperParams)
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    k = collect(range(k_hps.xmin, k_hps.xmax, length = k_hps.N))
    z = collect(range(z_hps.xmin, z_hps.xmax, length = z_hps.N))
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



k_steady_state_hi_StochasticSkiba(α::Real, A_H::Real, ρ::Real, δ::Real, κ::Real, stationary_mean::Real) = (α*A_H*stationary_mean/(ρ + δ))^(1/(1-α)) + κ
k_steady_state_lo_StochasticSkiba(α::Real, A_L::Real, ρ::Real, δ::Real, stationary_mean::Real) = (α*A_L*stationary_mean/(ρ + δ))^(1/(1-α))
k_star_Skiba(α::Real, A_H::Real, A_L::Real) = κ/(1-(A_L/A_H)^(1/α))
k_steady_state_hi(m::StochasticSkibaModel) = (m.α*m.A_H*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::StochasticSkibaModel) = (m.α*m.A_L*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))
k_star(m::StochasticSkibaModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))

y_H(m::StochasticSkibaModel) = (k, z) -> m.A_H*z*max(k - m.κ,0)^m.α
y_L(m::StochasticSkibaModel) = (k, z) -> m.A_L*z*k^m.α 

# Skiba production function
@inline function stochastic_skiba_production_function(k, z, α, A_H, A_L, κ)
    z .* max.(A_H .* pow.(max.(k .- κ, 0), α), A_L .* pow.(k, α))
end
# derivative of skiba production function
@inline function skiba_production_function_prime(k, z, α, A_H, A_L, κ)
    if k .> κ
        z .* A_H .* α .* pow.(k .- κ, α - 1)
    else
        z .* A_L .* α .* pow.(k, α - 1)
    end
end


@inline production_function(::StochasticSkibaModel, k, z, α::Real, A_H::Real, A_L::Real, κ::Real) = stochastic_skiba_production_function(k, z, α, A_H, A_L, κ)
@inline production_function(::StochasticSkibaModel, k, z, params::Vector) = stochastic_skiba_production_function(k, z, params[1], params[2], params[3], params[4])
@inline production_function(m::StochasticSkibaModel, k, z) = stochastic_skiba_production_function(k, z, m.α, m.A_H, m.A_L, m.κ)

@inline production_function_prime(::StochasticSkibaModel, k, z, α::Real, A_H::Real, A_L::Real, κ::Real) = skiba_production_function_prime(k, z, α, A_H, A_L, κ)
@inline production_function_prime(::StochasticSkibaModel, k, z, params::Vector) = skiba_production_function_prime(k, z, params[1], params[2], params[3], params[4])
@inline production_function_prime(m::StochasticSkibaModel, k, z) = skiba_production_function_prime(k, z, m.α, m.A_H, m.A_L, m.κ)





