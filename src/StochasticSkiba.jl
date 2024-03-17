

# %ORNSTEIN-UHLENBECK PROCESS dlog(z) = -the*log(z)dt + sig2*dW
# %STATIONARY DISTRIBUTION IS log(z) ~ N(0,Var) WHERE Var = sig2/(2*the)
# Var = 0.07;
# zmean = exp(Var/2); %MEAN OF LOG-NORMAL DISTRIBUTION N(0,Var)
# Corr = 0.9;
# the = -log(Corr);
# sig2 = 2*the*Var;

abstract type StochasticProcess end

struct OrnsteinUhlenbeckProcess <: StochasticProcess
    θ
    σ
    ρ
    stationary_σ
    zmean
end

function OrnsteinUhlenbeckProcess(; θ, σ)
    stationary_σ = σ^2/(2*θ)
    ρ = exp(-θ)
    zmean = exp(stationary_σ/2)
    OrnsteinUhlenbeckProcess(θ, σ, ρ, stationary_σ, zmean)
end


function from_stationary_OrnsteinUhlenbeckProcess(; ρ, stationary_σ)
    θ = -log(ρ)
    σ = sqrt(2*θ*stationary_σ) 
    zmean = exp(stationary_σ/2)
    OrnsteinUhlenbeckProcess(θ, σ, ρ, stationary_σ, zmean)
end


OrnsteinUhlenbeckProcess(θ = 1, σ = 2)
from_stationary_OrnsteinUhlenbeckProcess(ρ = 0.3678, stationary_σ = 2.0)
process_mean(p::OrnsteinUhlenbeckProcess) = exp(p.stationary_σ/2) 

struct StochasticSkibaModel{T <: Real} <: Model{T}
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
    stochasticprocess::StochasticProcess
end



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
    y = production_function(m, k)
    z = collect(range(z_hps.xmin, z_hps.xmax, length = z_hps.N))
    StateSpace((k = k, z = z), (y = y,))
end

function StochasticSkibaModel(; γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0, stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ =  0.1))
    StochasticSkibaModel(γ, α, ρ, δ, A_H, A_L, κ, stochasticprocess)
end



k_steady_state_hi_StochasticSkiba(α::Real, A_H::Real, ρ::Real, δ::Real, κ::Real, zmean::Real) = (α*A_H*zmean/(ρ + δ))^(1/(1-α)) + κ
k_steady_state_lo_StochasticSkiba(α::Real, A_L::Real, ρ::Real, δ::Real, zmean::Real) = (α*A_L*zmean/(ρ + δ))^(1/(1-α))
k_star_Skiba(α::Real, A_H::Real, A_L::Real) = κ/(1-(A_L/A_H)^(1/α))
k_steady_state_hi(m::StochasticSkibaModel) = (m.α*m.A_H*m.stochasticprocess.zmean/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::StochasticSkibaModel) = (m.α*m.A_L*m.stochasticprocess.zmean/(m.ρ + m.δ))^(1/(1-m.α))
k_star(m::SkibaModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))

y_H(m::SkibaModel) = (k) -> m.A_H*max(k - m.κ,0)^m.α
y_L(m::SkibaModel) = (k) -> m.A_L*k^m.α 

# Skiba production function
@inline function skiba_production_function(k, α, A_H, A_L, κ)
    max(A_H * pow(max(k - κ, 0), α), A_L * pow(k, α))
end
# derivative of skiba production function
@inline function skiba_production_function_prime(k, α, A_H, A_L, κ)
    if k > κ
        A_H * α * pow(k - κ, α - 1)
    else
        A_L * α * pow(k, α - 1)
    end
end


@inline production_function(::StochasticSkibaModel, k::Union{Real,Vector{<:Real}}, α::Real, A_H::Real, A_L::Real, κ::Real) = skiba_production_function.(k, α, A_H, A_L, κ)
@inline production_function(::StochasticSkibaModel, k::Union{Real,Vector{<:Real}}, params::Vector) = skiba_production_function.(k, params[1], params[2], params[3], params[4])
@inline production_function(m::StochasticSkibaModel, k::Union{Real,Vector{<:Real}}) = skiba_production_function.(k, m.α, m.A_H, m.A_L, m.κ)

@inline production_function_prime(::StochasticSkibaModel, k::Union{Real,Vector{<:Real}}, α::Real, A_H::Real, A_L::Real, κ::Real) = skiba_production_function_prime.(k, α, A_H, A_L, κ)
@inline production_function_prime(::StochasticSkibaModel, k::Union{Real,Vector{<:Real}}, params::Vector) = skiba_production_function_prime.(k, params[1], params[2], params[3], params[4])
@inline production_function_prime(m::StochasticSkibaModel, k::Union{Real,Vector{<:Real}}) = skiba_production_function_prime.(k, m.α, m.A_H, m.A_L, m.κ)





