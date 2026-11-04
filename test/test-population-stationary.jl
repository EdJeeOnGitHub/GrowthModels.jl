using Test, GrowthModels, SparseArrays, ForwardDiff, LinearAlgebra
@testset "Population defaults and explicit stationary mixtures" begin
    process=OrnsteinUhlenbeckProcess(θ=.1,σ=.1)
    make_model(shares)=FunctionalModel((k,z,a)->.6*k^.3*z*a,process,Population([.8,1.2],shares))
    model=make_model([.3,.7])
    grid=StateGrid(model;k=(n=50,bounds=(.01,8.)),z=(n=10,bounds=(.8,1.2)))
    fit=solve(model,grid)
    @test vec(sum(mass(stationary(fit));dims=(1,2))) ≈ [.3,.7]
    explicit=distribution(grid;masses=fill(1/length(grid),size(grid)))
    @test vec(sum(mass(stationary(fit,explicit));dims=(1,2))) ≈ [.5,.5]
    f=x->expectation(stationary(solve(make_model([x,1-x]),grid)),s->s.ability)
    @test ForwardDiff.derivative(f,.3) ≈ -.4 atol=1e-8
    # Direct mass-kernel checks use analytically known classes and absorption.
    small=StateGrid(model;k=[.1,.2],z=[.8,1.2])
    block=sparse([-1. 0 0 0;1 -1 0 0;0 1 -1 0;0 0 1 0])
    unique=kron(sparse(I,2,2),block)
    got=GrowthModels.population_stationary_mass(unique,small,Population([.8,1.2],[.3,.7]))
    @test got ≈ [0,0,0,.3,0,0,0,.7]
    ambiguous=copy(unique);ambiguous[5:8,5:8] .= 0
    @test_throws ArgumentError GrowthModels.population_stationary_mass(ambiguous,small,Population([.8,1.2],[.3,.7]))
    @test GrowthModels.population_stationary_mass(ambiguous,small,Population([.8,1.2],[1.,0.])) ≈ [0,0,0,1,0,0,0,0]
    initial=fill(1/8,8)
    @test GrowthModels.stationary_mass(ambiguous;initial_mass=initial) ≈ [0,0,0,.5,.125,.125,.125,.125]
end
