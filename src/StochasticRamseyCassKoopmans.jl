
function StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, θ, σ)
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end
#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function StateSpaceHyperParams(m::StochasticRamseyCassKoopmansModel; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40)
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

function StateSpace(m::StochasticRamseyCassKoopmansModel, statespacehyperparams::StateSpaceHyperParams)
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    k = collect(range(k_hps.xmin, k_hps.xmax, length = k_hps.N))
    z = collect(range(z_hps.xmin, z_hps.xmax, length = z_hps.N))
    # z' creates Nk x Nz matrix
    y = production_function(m, k, z')
    StateSpace((k = k, z = z), (y = y,))
end


function StochasticRamseyCassKoopmansModel(
    stochasticprocess::Union{StochasticProcess,Nothing}; 
     γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6)
    if isnothing(stochasticprocess)
        stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ =  0.1)
    end
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, stochasticprocess)
end


function StochasticRamseyCassKoopmansModel(;
    γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6, θ = -log(0.9), σ = 0.1)
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end

function StochasticRamseyCassKoopmansModel(
    γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6, θ = -log(0.9), σ = 0.1)
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end

k_steady_state_hi_StochasticRamseyCassKoopmans(α::Real, A::Real, ρ::Real, δ::Real, stationary_mean::Real) = (α*A*stationary_mean/(ρ + δ))^(1/(1-α))
k_steady_state_lo_StochasticRamseyCassKoopmans(α::Real, A::Real, ρ::Real, δ::Real, stationary_mean::Real) = (α*A*stationary_mean/(ρ + δ))^(1/(1-α))
k_star_StochasticRamseyCassKoopmans(α::Real, A::Real, ρ::Real, δ::Real, stationary_mean::Real) = (α*A*stationary_mean/(ρ + δ)) ^ (1/(1-α)) 
k_steady_state_hi(m::StochasticRamseyCassKoopmansModel) = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) 
k_steady_state_lo(m::StochasticRamseyCassKoopmansModel) = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) 
k_star(m::StochasticRamseyCassKoopmansModel) = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))



# Production function
@inline function stochastic_rck_production_function(k, z, α, A)
     A .* z .* pow.(k, α)
end
# Derivative of production function
@inline function stochastic_rck_production_function_prime(k, z, α, A)
     A .* z .* α .* pow.(k, α - 1)
end

@inline production_function(::StochasticRamseyCassKoopmansModel, k, z, α::Real, A::Real) = stochastic_rck_production_function(k, z, α, A)
@inline production_function(::StochasticRamseyCassKoopmansModel, k, z, params::Vector) = stochastic_rck_production_function(k, z, params[1], params[2])
@inline production_function(m::StochasticRamseyCassKoopmansModel, k, z) = stochastic_rck_production_function(k, z, m.α, m.A)

@inline production_function_prime(::StochasticRamseyCassKoopmansModel, k, z, α::Real, A::Real, δ::Real) = stochastic_rck_production_function_prime(k, z, α, A)
@inline production_function_prime(::StochasticRamseyCassKoopmansModel, k, z, params::Vector) = stochastic_rck_production_function_prime(k, z, params[1], params[2])
@inline production_function_prime(m::StochasticRamseyCassKoopmansModel, k, z) = stochastic_rck_production_function_prime(k, z, m.α, m.A)


function plot_production_function(m::StochasticRamseyCassKoopmansModel, k, z)
    y = production_function(m, collect(k), collect(z)')
    plot(k, y, label="")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end