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
    discrete=DiscreteSkibaModel()
    discrete_fit=solve(discrete,StateGrid(discrete;k=(n=60,));history=true)
    @test plot_model(discrete_fit) isa Plots.Plot
    @test plot_diagnostics(discrete_fit) isa Plots.Plot
    @test_throws ArgumentError plot_diagnostics(solve(discrete,discrete_fit.grid))

end
