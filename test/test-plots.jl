using Test, GrowthModels
@testset "Optional plotting extension" begin
    @test Base.get_extension(GrowthModels,:GrowthModelsPlotsExt) === nothing
    @eval import Plots
    @test Base.get_extension(GrowthModels,:GrowthModelsPlotsExt) !== nothing
    model=RamseyCassKoopmansModel()
    fit=solve(model,StateGrid(model;k=(n=60,));history=true)
    @test plot_model(fit) isa Plots.Plot
    @test plot_diagnostics(fit) isa Plots.Plot
    @test_throws ArgumentError plot_diagnostics(solve(model,fit.grid))
end
