
#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function StateSpaceHyperParams(m::StochasticRamseyCassKoopmansModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
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

function StateSpaceHyperParams(m::StochasticRamseyCassKoopmansModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40) where {T <: Real, S <: PoissonProcess}
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin)
    # z_hps
    zmin = minimum(m.stochasticprocess.z)
    zmax = maximum(m.stochasticprocess.z)
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps))
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

function StateSpace(m::StochasticRamseyCassKoopmansModel{T, S}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
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



function StochasticRamseyCassKoopmansModel(
    γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A = 0.6, θ = -log(0.9), σ = 0.1)
    StochasticRamseyCassKoopmansModel(γ, α, ρ, δ, A, OrnsteinUhlenbeckProcess(θ = θ, σ = σ))
end


# OrnsteinUhlenbeckProcess
k_steady_state_hi(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) 
k_steady_state_lo(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: OrnsteinUhlenbeckProcess} = (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α)) 
# Poisson
k_steady_state_hi(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.α*m.A/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)
k_steady_state_lo(m::StochasticRamseyCassKoopmansModel{T, S}) where {T <: Real, S <: PoissonProcess} = (m.α*m.A/(m.ρ + m.δ))^(1/(1-m.α))  + process_mean(m.stochasticprocess)

k_star(m::StochasticRamseyCassKoopmansModel{T, S})  where {T <: Real, S <: OrnsteinUhlenbeckProcess}= (m.α*m.A*process_mean(m.stochasticprocess)/(m.ρ + m.δ))^(1/(1-m.α))
k_star(m::StochasticRamseyCassKoopmansModel{T, S})  where {T <: Real, S <: PoissonProcess} = (m.α*m.A/(m.ρ + m.δ))^(1/(1-m.α)) + process_mean(m.stochasticprocess)





# Production function
@inline function stochastic_rck_production_function(::OrnsteinUhlenbeckProcess, k, z, α, A)
     A .* z .* k .^ α
end
# Poisson
@inline function stochastic_rck_production_function(::PoissonProcess, k, z, α, A)
     A .* k .^ α .+ z
end
# Derivative of production function
@inline function stochastic_rck_production_function_prime(::OrnsteinUhlenbeckProcess, k, z, α, A)
     A .* z .* α .* k .^ (α - 1)
end
# Poisson
@inline function stochastic_rck_production_function_prime(::PoissonProcess, k, z, α, A)
     A .* α .* k .^ (α - 1)
end

@inline production_function(::StochasticRamseyCassKoopmansModel{T, S}, k, z, α::Real, A::Real) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function(S(), k, z, α, A)
@inline production_function(::StochasticRamseyCassKoopmansModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function(S(), k, z, params[1], params[2])
@inline production_function(m::StochasticRamseyCassKoopmansModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function(m.stochasticprocess, k, z, m.α, m.A)

@inline production_function_prime(::StochasticRamseyCassKoopmansModel{T, S}, k, z, α::Real, A::Real, δ::Real) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function_prime(S(), k, z, α, A)
@inline production_function_prime(::StochasticRamseyCassKoopmansModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function_prime(S(), k, z, params[1], params[2])
@inline production_function_prime(m::StochasticRamseyCassKoopmansModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_rck_production_function_prime(m.stochasticprocess, k, z, m.α, m.A)


function plot_production_function(m::StochasticRamseyCassKoopmansModel, k, z)
    y = production_function(m, collect(k), collect(z)')
    plot(k, y, label="")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end