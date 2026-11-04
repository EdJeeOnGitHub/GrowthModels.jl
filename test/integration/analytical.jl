# Exercise the shared downstream module without running its large figure jobs.
using Test, GrowthModels, DataFrames, LinearAlgebra
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))
include(joinpath(analysis_root, "code/growth-models/TrapProbabilities.jl"))
using .TrapProbabilities

@testset "Analytical exports and probability summaries" begin
    fit = setup_simple_model(; Nk=80, Nz=5, Nη=3)
    @test fit.converged
    output = create_output_data(fit)
    @test nrow(output) == length(fit.value)
    @test output.k == vec(coordinates(fit.grid).k)
    @test output.c == vec(fit.policy.consumption)
    @test output.kdot == vec(fit.policy.drift)
    @test output.eta == vec(coordinates(fit.grid).ability)
    initial = distribution(fit.grid; masses=fill(1/length(fit.value), size(fit.grid)))
    final = mass(advance(Evolution(fit; dt=1.), initial, 12.))
    @test sum(final) ≈ 1
    thresholds = [2.5, 3.0]
    ps, joint = create_expost_prob_grid(final, thresholds, coordinates(fit.grid).k)
    @test all(0 .<= ps .<= 1)
    @test vec(sum(joint; dims=2)) ≈ ps
    @test nrow(create_p_df(fit, thresholds, ps, joint)) == 8
    backward = create_exante_prob_grid(fit, 2.5, 12)
    # Independent forward/backward duality checks generator orientation and dt.
    @test sum(mass(initial) .* backward) ≈ ps[1] atol=1e-10
    @test minimum(backward) >= -1e-12 && maximum(backward) <= 1+1e-12
    node = fit.grid.k[30]
    at_node = create_exante_prob_grid(fit, node, 0)
    @test at_node == (coordinates(fit.grid).k .>= node)
end
