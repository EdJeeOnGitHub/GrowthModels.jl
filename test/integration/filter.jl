using Test, GrowthModels, StochasticEM, Random
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))
include(joinpath(analysis_root, "man/notebook/filter/SimulatePanel.jl"))
using .SimulatePanel
using .SimulatePanel.AnalysisPolicy

@testset "Filter structural simulation migration" begin
    basis = StochasticEM.BasisFunctions.PolynomialBasis{Float64}(1)
    for constructor in (RCKSMSimParams, SkibaSMSimParams)
        Random.seed!(91)
        params = constructor(91,basis; ρ=.7,σ_ξ=.1,σ_η=.1,σ_ε=0.,σ=0.)
        @test params.sm.converged
        @test size(params.sm.grid) == (500,10,10)
        @test first(params.sm.grid.k) == .01
        _, _, data, _, production_fn = simulate_panel_data(30,3,params; seed=42)
        @test size(data.k,1) > 0 && size(data.k,2) == 3
        @test all(isfinite,data.c) && all(isfinite,data.k)
        @test data.k[:,2:end] ≈ data.y_all[:,1:end-1] .- data.c[:,1:end-1] .+
            params.delta_k .* data.k[:,1:end-1]
        @test production_fn(data.k[1,1],data.eta[1],data.omega[1,1]) ≈ data.y_all[1,1]
        k, v = capital_profile(params.sm,:value)
        _, forward = capital_profile(params.sm,:forward)
        _, backward = capital_profile(params.sm,:backward)
        @test forward[1:end-1] ≈ diff(v)./diff(k)
        @test backward[2:end] ≈ diff(v)./diff(k)
        @test backward[1] > 0 && forward[end] > 0
        surface = analysis_surface(params.sm,:forward)
        z, ability = params.sm.grid.z[5],params.sm.grid.ability[5]
        @test surface(k[20],z,ability) ≈ forward[20]
        @test_throws BoundsError surface(k[20],z,2.)
    end
end
