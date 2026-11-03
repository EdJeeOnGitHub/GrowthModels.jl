# Run with the migrated root environment, or its joint-generative-model environment.
using Test, GrowthModels
const analysis_root=normpath(joinpath(@__DIR__,"../../../aggregate-transfers"))
include(joinpath(analysis_root,"sims/mainline/StructuralDGP.jl"))
include(joinpath(analysis_root,"sims/mainline/StructuralCalibration.jl"))
include(joinpath(analysis_root,"sims/mainline/StructuralSolutionValidation.jl"))
@testset "Mainline structural analysis API migration" begin
    p=StructuralDGP.Parameters("rck",1.,.6,.3,1.5,2.,.1,.05,.05,2.,.1,.1,.1,.1,.01,8.,80)
    eta=[.8,1.2];z=collect(range(.8,1.2;length=8))
    fit=StructuralDGP.solve_on_support(p;eta=eta,z=z)
    @test fit isa GrowthSolution && fit.converged
    t=collect(range(0.,1.;length=p.nk));w=t.+100 .* t.^3
    @test fit.grid.k ≈ p.kmin .+ (p.kmax-p.kmin).*w./last(w)
    @test StructuralDGP.solved_grids(fit).eta == eta
    @test StructuralDGP.interpolated_consumption(fit,1.,1.,1.) ≈
        .5consumption(fit,(k=1.,z=1.,ability=.8))+.5consumption(fit,(k=1.,z=1.,ability=1.2))
    @test_throws ErrorException StructuralDGP.interpolated_consumption(fit,1.,1.,1.5)
    @test StructuralSolutionValidation.validate(fit,p).passed
    kernel=StructuralDGP.StructuralIntegration.shock_kernel(p;grid=z)
    @test kernel.Q ≈ Matrix(generator(process(fit.model),z))
    variance=Dict("ability_grid"=>Dict("lower"=>.8,"upper"=>1.2,"nodes"=>2),"lower"=>.8,"upper"=>1.2,"nodes"=>8)
    @test StructuralCalibration.solve_model(p,variance).value ≈ fit.value
end
