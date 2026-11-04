using Test, GrowthModels, LinearAlgebra
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))
const notebook = joinpath(analysis_root, "man/notebook/metastability/pcca.jl")
# Load the actual notebook solver without running its interactive figure cells.
for expr in Meta.parseall(read(notebook, String); filename=notebook).args
    expr isa Expr && expr.head == :function && Core.eval(@__MODULE__, expr)
end
include(joinpath(dirname(notebook), "PCCA.jl"))
using .PCCA

@testset "Metastability notebook public API" begin
    fit = setup_simple_model(; Nk=60, Nz=6)
    @test fit.converged
    t = collect(range(0., 1.; length=60))
    @test fit.grid.k ≈ 1e-3 .+ (10. - 1e-3) .* (t .+ 5 .* t.^10) ./ 6
    P = convert_generator_to_matrix(copy(fit.generator))
    @test maximum(abs.(sum(P; dims=2) .- 1)) < 1e-10
    @test minimum(P) >= -1e-12
    chi, sets, eigenvals = pcca_plus(P, 2)
    @test size(chi) == (length(fit.value), 2)
    @test maximum(abs.(sum(chi; dims=2) .- 1)) < 1e-8
    @test sort(vcat(sets...)) == collect(1:length(fit.value))
    @test length(vec(fit.policy.drift)[sets[1]]) == length(sets[1])
end
