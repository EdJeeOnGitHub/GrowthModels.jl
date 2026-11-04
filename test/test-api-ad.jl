using Test, GrowthModels, ForwardDiff, SparseArrays, LinearAlgebra

@testset "Differentiation contracts" begin
    grid=StateGrid(RamseyCassKoopmansModel();k=(n=50,bounds=(.1,5.)),)
    function moments(a; warm=nothing, moving=false)
        m=RamseyCassKoopmansModel(A=a)
        g=moving ? StateGrid(m;k=(n=50,)) : grid
        fit=solve(m,g;initial=warm,value_tolerance=1e-9,residual_tolerance=1e-9)
        require_converged(fit)
        d=distribution(g;masses=fill(1/length(g),size(g)))
        e=Evolution(fit;dt=.1)
        return [consumption(fit,(k=1.,)),expectation(advance(e,d,1.03),s->s.k),expectation(stationary(fit,d),s->s.k)]
    end
    h=1e-5
    for moving in (false,true)
        f=a->moments(a;moving=moving)
        ad=ForwardDiff.derivative(f,.6)
        fd=(f(.6+h)-f(.6-h))/(2h)
        @test ad ≈ fd rtol=2e-4 atol=1e-7
    end
    warm=solve(RamseyCassKoopmansModel(),grid;value_tolerance=1e-9,residual_tolerance=1e-9)
    @test ForwardDiff.derivative(a->moments(a;warm=warm),.6) ≈ ForwardDiff.derivative(moments,.6) rtol=1e-5
    # Differentiate a probability input against a fixed real generator/factor.
    e=Evolution(warm;dt=.1)
    f=x->begin
        p=fill(zero(x),size(grid));p[10]=x;p[20]=1-x
        expectation(advance(e,distribution(grid;masses=p),1.),s->s.k)
    end
    @test ForwardDiff.derivative(f,.4) ≈ (f(.4+h)-f(.4-h))/(2h) rtol=1e-6
    # Internal solve supports nested duals, without adding methods to Base.\.
    sparsef=p->GrowthModels.solve_sparse_factor(GrowthModels.factor_sparse(sparse([3+p[1] -1.;-1. 2+p[2]])),[p[1],p[2]])
    p=[.2,.3]
    @test ForwardDiff.jacobian(sparsef,p) ≈ hcat([(sparsef(p+h.*[i==j for i=1:2])-sparsef(p-h.*[i==j for i=1:2]))/(2h) for j=1:2]...) rtol=1e-6
    H=ForwardDiff.hessian(p->sum(sparsef(p)),p)
    grad=p->ForwardDiff.gradient(p->sum(sparsef(p)),p)
    @test H ≈ hcat([(grad(p+h.*[i==j for i=1:2])-grad(p-h.*[i==j for i=1:2]))/(2h) for j=1:2]...) rtol=1e-5
end

@testset "Volatility, resources and regridded initialization derivatives" begin
    fixed_model=StochasticRamseyCassKoopmansModel()
    fixed_grid=StateGrid(fixed_model;k=(n=45,bounds=(.1,5.)),z=(n=9,bounds=(.6,1.5)))
    for moving in (false,true)
        function moment(sigma)
            process=OrnsteinUhlenbeckProcess(θ=.1,σ=sigma)
            model=StochasticRamseyCassKoopmansModel(process)
            grid=moving ? StateGrid(model;k=(n=45,bounds=(.1,5.)),z=(n=9,width=3)) : fixed_grid
            fit=solve(model,grid;value_tolerance=1e-9,residual_tolerance=1e-9)
            require_converged(fit)
            d=distribution(grid;masses=fill(1/length(grid),size(grid)))
            [expectation(advance(Evolution(fit;dt=.1),d,.35),s->s.k),
             expectation(stationary(fit,d),s->s.k)]
        end
        h=1e-5
        @test ForwardDiff.derivative(moment,.1) ≈ (moment(.1+h)-moment(.1-h))/(2h) rtol=5e-4 atol=1e-6
    end
    base=RamseyCassKoopmansModel()
    oldgrid=StateGrid(base;k=(n=60,bounds=(.05,6.)))
    oldfit=solve(base,oldgrid;value_tolerance=1e-9,residual_tolerance=1e-9)
    function floor_moment(transfer;warm=false)
        model=ResourceModel(base;transfer=transfer,income_floor=.35)
        grid=StateGrid(model;k=(n=45,bounds=(.1,5.)))
        fit=solve(model,grid;initial=warm ? regrid(oldfit,grid) : nothing,
            value_tolerance=1e-9,residual_tolerance=1e-9)
        require_converged(fit)
        consumption(fit,(k=1.,))
    end
    h=1e-5
    cold=ForwardDiff.derivative(floor_moment,.1)
    @test cold ≈ (floor_moment(.1+h)-floor_moment(.1-h))/(2h) rtol=1e-4
    @test ForwardDiff.derivative(x->floor_moment(x;warm=true),.1) ≈ cold rtol=1e-4
end

@testset "Poisson rate derivatives on a fixed grid" begin
    process=PoissonProcess(z=[0.,.1],Q=[-.2 .2;.3 -.3])
    base=StochasticRamseyCassKoopmansModel(process)
    grid=StateGrid(base;k=(n=50,bounds=(.1,5.)))
    function moment(rate)
        p=PoissonProcess(z=[0.,.1],Q=[-rate rate;.3 -.3])
        fit=solve(StochasticRamseyCassKoopmansModel(p),grid;
            value_tolerance=1e-9,residual_tolerance=1e-9)
        expectation(stationary(fit),s->s.k)
    end
    h=1e-5
    @test ForwardDiff.derivative(moment,.2) ≈ (moment(.2+h)-moment(.2-h))/(2h) rtol=2e-4 atol=1e-7
end
