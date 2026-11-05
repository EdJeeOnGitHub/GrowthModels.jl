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

@testset "Stationary probabilities tolerate solver roundoff" begin
    # Tail probabilities are near machine precision here, so a direct solve
    # returns tiny negative entries for these valid generators.
    model=StochasticRamseyCassKoopmansModel(OrnsteinUhlenbeckProcess(θ=.1,σ=.05))
    for nz in (11,40)
        grid=StateGrid(model;k=(n=200,),z=(n=nz,))
        fit=solve(model,grid)
        @test fit.converged
        invariant=stationary(fit)
        @test all(>=(0),mass(invariant))
        @test sum(mass(invariant)) ≈ 1 atol=1e-13
        @test maximum(abs,GrowthModels.forward_generator(fit)*vec(mass(invariant))) < 1e-9
    end
    # Roundoff is removed; a materially negative solve remains an error.
    cleaned=GrowthModels.clip_probability_roundoff!([.5,-1e-16,.5])
    @test cleaned[2] == 0
    @test sum(cleaned) ≈ 1 atol=1e-15
    @test_throws ArgumentError GrowthModels.clip_probability_roundoff!([.6,-.1,.5])
    # Results without negative entries are returned exactly as solved.
    @test GrowthModels.clip_probability_roundoff!([.3,.3,.3]) == [.3,.3,.3]
    # Clipping follows the primal value and keeps a usable tangent elsewhere.
    derivative=ForwardDiff.derivative(0.5) do x
        GrowthModels.clip_probability_roundoff!([x,-1e-16*x,1-x])[1]
    end
    @test derivative ≈ 1 atol=1e-12
end

@testset "Metastable states fail loudly, not silently" begin
    # Low-ability types can escape the low state only through a rare productivity path,
    # so the transient solve is singular to machine precision.
    model=ResourceModel(StochasticSkibaAbilityModel(OrnsteinUhlenbeckProcess(θ=-log(.9),σ=.1);
        A_L=.4,A_H=.9,α=.5,κ=1.,δ=.25,ρ=.05);population=Population(collect(range(.5,1.5;length=40)),fill(1/40,40)))
    grid=StateGrid(model;k=(n=150,bounds=(.02,30.),spacing=:stretched),z=(n=15,bounds=(.9,1.1)))
    initial=distribution(grid;masses=fill(1/length(grid),size(grid)))
    fit=solve(model,grid)
    err=try stationary(fit,initial); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("metastable",sprint(showerror,err))
    # The suggested finite horizon works and keeps each type's mass.
    d=advance(Evolution(fit;dt=1.0),initial,100.0)
    @test vec(sum(mass(d);dims=(1,2))) ≈ fill(1/40,40) rtol=1e-8
    # Transient mass is still absorbed into the class it leads to.
    block=sparse([-1. 0 0;1. 0 0;0 0 0])
    @test GrowthModels.stationary_mass(block;initial_mass=[.5,.25,.25]) ≈ [0,.75,.25]
end
