
struct SkibaModel <: Model
    γ::Real
    α::Real
    ρ::Real
    δ::Real
    A_H::Real
    A_L::Real
    κ::Real
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

struct SmoothSkibaModel <: Model
    γ::Real
    α::Real
    ρ::Real
    δ::Real
    A_H::Real
    A_L::Real
    κ::Real
    β::Real
end


struct RamseyCassKoopmansModel <: Model
    γ::Real
    α::Real
    ρ::Real
    δ::Real
    A::Real
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
