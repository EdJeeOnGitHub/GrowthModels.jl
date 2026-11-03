using Test, GrowthModels, SparseArrays, LinearAlgebra
@testset "Generator and limiting-class invariants" begin
    # Two absorbing classes, one transient source with a known 1:2 split.
    A=sparse([0. 0. 1.;0. 0. 2.;0. 0. -3.])
    p=GrowthModels.StationaryDistribution(A,ones(3);initial_density=[0.,0.,1.])
    @test p ≈ [1/3,2/3,0.]
    @test_throws ArgumentError GrowthModels.StationaryDistribution(A,ones(3))
    @test_throws GrowthModels.ValueFunctionError GrowthModels.check_generator(sparse([-1. 1.;-1. 1.]))
    for sigma in (0.,.01,.1), n in (5,21)
        m=StochasticRamseyCassKoopmansModel(OrnsteinUhlenbeckProcess(θ=.1,σ=sigma))
        # Explicit bounds also support the deterministic zero-volatility limit.
        g=StateGrid(m;k=[.1,.5,1.],z=collect(range(.8,1.2;length=n)))
        _,_,Q=GrowthModels.numerical_state(m,g)
        @test maximum(abs.(sum(Q;dims=2))) < 1e-10
        rows,cols,values=findnz(Q)
        @test all(values[rows .!= cols] .>= 0)
        z=vec(coordinates(g).z)
        drift=(-.1 .* log.(z) .+ sigma^2/2) .* z
        qz=Q*z
        @test qz[4:end-3] ≈ drift[4:end-3] atol=1e-10
    end
    for model in (RamseyCassKoopmansModel(),SkibaModel())
        g=StateGrid(model;k=(n=300,spacing=:stretched,power=2.))
        fit=solve(model,g)
        @test fit.converged
        @test fit.generator * vec(g.k) ≈ vec(fit.policy.drift) atol=1e-9
        @test maximum(abs.(sum(fit.generator;dims=2))) < 1e-9
        allcells=distribution(g;masses=fill(1/length(g),size(g)))
        finaldist=advance(Evolution(fit;dt=.1),allcells,10.)
        @test sum(mass(finaldist)) ≈ 1 atol=1e-11
    end
end

@testset "Hamiltonian selection and structured linear solves" begin
    vf=[.25,4.,2.25,2.25,.25,4.]
    vb=[.25,.25,.01,.25,4.,4.]
    choices=[GrowthModels.upwind_choice(vf[i],vb[i],1.,2.,i,6) for i in 1:6]
    @test first.(choices) ≈ [1.,.5,10.,2/3,1.,1.]
    @test all(c->sum(c[2:4])==1,choices)
    @test all(c->c[2]*c[5]+c[3]*c[6] ≈ 1-c[1],choices)
    A=sparse([3. -1 0;-.5 2 -.2;0 -.3 1.])
    b=[1.,2.,3.]
    tri=GrowthModels.TridiagonalHJBSolver(A)
    x=zeros(3);GrowthModels.solve_linear!(x,tri,b)
    @test x ≈ A\b
    block=kron(sparse(I,2,2),A)
    solver=GrowthModels.SparseBlockSolver(block;blocksize=3)
    rhs=collect(1.:6.);out=zeros(6)
    GrowthModels.solve_linear!(out,solver,rhs)
    @test out ≈ block\rhs
    coupled=copy(block);coupled[3,4]=-.1
    @test GrowthModels.independent_blocksize(coupled,3)==6
    @test_throws ArgumentError GrowthModels.refactor!(solver,coupled)
end

@testset "Boundary constraints and process sampling" begin
    process=PoissonProcess(z=[.8,1.,1.2],Q=[-1. 1 0;1 -2 1;0 1 -1])
    types=Population([1.],[1.])
    model=FunctionalModel((k,z,a)->z==1 ? -.1 : 1.,process,types;γ=1.5)
    grid=StateGrid(model;k=[.1,1.])
    @test_throws DomainError solve(model,grid)
    high=RamseyCassKoopmansModel(γ=1.5)
    fit=solve(high,StateGrid(high;k=(n=80,bounds=(.1,100.))))
    @test fit.converged
    @test fit.policy.drift[end] < 0
    p=OrnsteinUhlenbeckProcess(θ=.1,σ=.1)
    for initial in (0.,[0.,.1])
        times,values=sample(p,initial,1.,.3;seed=1)
        @test length(times)==(initial isa Number ? length(values) : size(values,2))
    end
    times,values=sample(p,[0.,.1],1.,.3,3;seed=1)
    @test length(times)==size(values,2)
end
