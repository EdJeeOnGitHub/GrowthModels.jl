

struct StochasticTwoAssetKinkedModel{T <: Real, S <: StochasticProcess} <: StochasticModel{T, S}
    γ::T
    r_a::T
    r_b_pos::T
    r_b_neg::T
    ρ::T
    χ_0::T
    χ_1::T
    ξ::T
    w::T
    stochasticprocess::S
end

function StochasticTwoAssetKinkedModel(stochasticprocess::Union{StochasticProcess,Nothing};
            γ = 2.0, r_a = 0.05, r_b_pos = 0.03, r_b_neg = 0.12, ρ = 0.06, χ_0 = 0.03, χ_1 = 2, ξ = 0.1, w = 4)
    if isnothing(stochasticprocess)
        stochasticprocess = PoissonProcess(z = [0.8, 1.3], λ = [1/3, 1/3])
    end
    StochasticTwoAssetKinkedModel(γ, r_a, r_b_pos, r_b_neg, ρ, χ_0, χ_1, ξ, w, stochasticprocess)
end

#### Steady State Helpers ------------------------------------------------------
# Skipping this
#### Production Function -------------------------------------------------------
# PoissonProcess just an income shock
@inline function stochastic_two_asset_fa(::Union{PoissonProcess,Type{PoissonProcess}}, a, z, r_a, ξ, w)
    r_a .* a .+ ξ .* w .* z 
end

@inline function stochastic_two_asset_fb(::Union{PoissonProcess,Type{PoissonProcess}}, b, z, r_b_pos, r_b_neg, ξ, w)
    r_b = (b .< 0) .* r_b_neg .+ (b .>= 0) .* r_b_pos
    return (1 .- ξ) .* w .* z .+ r_b .* b 
end

function cost_adjustment(d, a, χ_0, χ_1) 
    χ_0 .* abs.(d) + 0.5 .* χ_1 .* a .* (d ./ a).^2
end


# Method dispatch for various forms of arguments
# k, z, and b + params passed directly
@inline production_function(::StochasticSkibaDebtModel{T,S}, k, z, b, α::Real, A_H::Real, A_L::Real, κ::Real) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function(S, k, z, b, α, A_H, A_L, κ)
@inline production_function_prime(::StochasticSkibaDebtModel{T, S}, k, z, b, α::Real, A_H::Real, A_L::Real, κ::Real) where {T <: Real, S <: StochasticProcess} =  stochastic_skiba_debt_production_function_prime(S, k, z, b, α, A_H, A_L, κ)

# k, z, b, and params passed as a vector
@inline production_function(::StochasticSkibaDebtModel{T, S}, k, z, b, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function(S, k, z, b, params[1], params[2], params[3], params[4])
@inline production_function_prime(::StochasticSkibaDebtModel{T, S}, k, z, params::Vector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function_prime(S, k, z, b, params[1], params[2], params[3], params[4])
# k, z, and params passed using model fields
@inline production_function(m::StochasticSkibaDebtModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function(m.stochasticprocess, k, z, b, m.α, m.A_H, m.A_L, m.κ)
@inline production_function_prime(m::StochasticSkibaDebtModel{T, S}, k, z) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function_prime(m.stochasticprocess, k, z, b, m.α, m.A_H, m.A_L, m.κ)
# k, z, b passed as matrix and params using vector
@inline production_function(::StochasticSkibaDebtModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function(S, x[:,1], x[:,2], x[:,3], params[1], params[2], params[3], params[4])
@inline production_function_prime(::StochasticSkibaDebtModel{T, S}, x::AbstractMatrix, params::AbstractVector) where {T <: Real, S <: StochasticProcess} = stochastic_skiba_debt_production_function_prime(S, x[:,1], x[:,2], x[:, 3], params[1], params[2], params[3], params[4])


