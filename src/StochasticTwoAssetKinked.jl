

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
            γ = 2.0, r_a = 0.05, r_b_pos = 0.03, r_b_neg = 0.12, ρ = 0.06, χ_0 = 0.03, χ_1 = 2.0, ξ = 0.1, w = 4.0)
    if isnothing(stochasticprocess)
        stochasticprocess = PoissonProcess(z = [0.8, 1.3], λ = [1/3, 1/3])
    end
    StochasticTwoAssetKinkedModel(γ, r_a, r_b_pos, r_b_neg, ρ, χ_0, χ_1, ξ, w, stochasticprocess)
end

#### Steady State Helpers ------------------------------------------------------
# Skipping this

function StateSpaceHyperParams(m::StochasticTwoAssetKinkedModel{T, S}; Nb = 100,  bmin = -2, bmax = 40, Na = 50, amin = 0, amax = 70, Nz = 2) where {T <: Real, S <: PoissonProcess}
    b_hps = HyperParams(N = Nb, xmax = bmax, xmin = bmin)
    a_hps = HyperParams(N = Na, xmax = amax, xmin = amin)
    # z_hps
    zmin = minimum(m.stochasticprocess.z)
    zmax = maximum(m.stochasticprocess.z)
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    return StateSpaceHyperParams((b = b_hps, a = a_hps, z = z_hps))
end

function StateSpace(m::StochasticTwoAssetKinkedModel{T, S}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: PoissonProcess}
    b_hps = statespacehyperparams[:b]
    a_hps = statespacehyperparams[:a]
    z_hps = statespacehyperparams[:z]

    b = collect(range(b_hps.xmin, b_hps.xmax, length = b_hps.N))
    a = collect(range(a_hps.xmin, a_hps.xmax, length = a_hps.N))
    z = collect(range(z_hps.xmin, z_hps.xmax, length = z_hps.N))
    z = vcat(z_hps.xmin, z_hps.xmax)
    # z' creates Na x Nz matrix
    y_a = stochastic_two_asset_fa(m.stochasticprocess, a, z', m.r_a, m.ξ, m.w)
    y_b = stochastic_two_asset_fb(m.stochasticprocess, b, z', m.r_b_pos, m.r_b_neg, m.ξ, m.w)
    StateSpace((b = b, a = a, z = z), (y_a = y_a, y_b = y_b))
end

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




