
struct SkibaModel{T <: Real} <: Model{T}
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
end

# struct StochasticSkibaModel <: Model
#     γ::Real
#     α::Real
#     ρ::Real
#     δ::Real
#     A_H::Real
#     A_L::Real
#     κ::Real
#     σ_z::Real
# end

struct SmoothSkibaModel{T <: Real} <: Model{T} 
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
    β::T
end


struct RamseyCassKoopmansModel{T <: Real} <: Model{T} 
    γ::T
    α::T
    ρ::T
    δ::T
    A::T
end

function show(io::IO, m::SkibaModel)
    print(io, "SkibaModel: γ = ", m.γ, ", α = ", m.α, ", ρ = ", m.ρ, ", δ = ", m.δ, ", A_H = ", m.A_H, ", A_L = ", m.A_L, ", κ = ", m.κ)
end

function show(io::IO, m::SmoothSkibaModel)
    print(io, "SmoothSkibaModel: γ = ", m.γ, ", α = ", m.α, ", ρ = ", m.ρ, ", δ = ", m.δ, ", A_H = ", m.A_H, ", A_L = ", m.A_L, ", κ = ", m.κ, ", β = ", m.β)
end

function show(io::IO, m::RamseyCassKoopmansModel)
    print(io, "RamseyCassKoopmansModel: γ = ", m.γ, ", α = ", m.α, ", ρ = ", m.ρ, ", δ = ", m.δ, ", A = ", m.A)
end

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

struct StochasticRamseyCassKoopmansModel{T <: Real} <: Model{T}
    γ::T
    α::T
    ρ::T
    δ::T
    A::T
    stochasticprocess::StochasticProcess
end

