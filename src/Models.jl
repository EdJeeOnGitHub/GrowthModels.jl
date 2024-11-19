
struct SkibaModel{T <: Real} <: DeterministicModel{T}
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
end

struct SmoothSkibaModel{T <: Real} <: DeterministicModel{T} 
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
    β::T
end


struct RamseyCassKoopmansModel{T <: Real} <: DeterministicModel{T} 
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

struct StochasticRamseyCassKoopmansModel{T <: Real, S <: StochasticProcess} <: StochasticModel{T, S}
    γ::T
    α::T
    ρ::T
    δ::T
    A::T
    stochasticprocess::S
end


struct StochasticSkibaModel{T <: Real, S <: StochasticProcess} <: StochasticModel{T, S}
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
    stochasticprocess::S
end


struct StochasticSkibaAbilityModel{T <: Real, S <: StochasticProcess} <: StochasticModel{T, S}
    γ::T
    α::T
    ρ::T
    δ::T
    A_H::T
    A_L::T
    κ::T
    η_mean::T
    stochasticprocess::S
end