using Test, GrowthModels, Random
const DCGM = GrowthModels

const DC_BASE = DiscreteSkibaModel(productivity=LogAR1(persistence=0.9,innovation_std=0.1))
dc_grid(m;n=150,hi=25.0,nz=5) = StateGrid(m;k=(n=n,bounds=(0.05,hi),spacing=:stretched),z=(n=nz,))

@testset "Discrete credit: no lending premium reproduces the base model" begin
    # With θ=0 and a rate just above -δ, investing beats lending, so k′ = s and the
    # problem in cash on hand x = y(k,z) + (1-δ)k is the base model's problem in k.
    m = DiscreteCreditModel(DC_BASE;θ=0.0,r=-DC_BASE.depreciation+1e-6)
    @test leverage(m) == 1
    fit = require_converged(solve(m,dc_grid(m;n=300)))
    base = require_converged(solve(DC_BASE,StateGrid(DC_BASE;k=(n=300,bounds=(0.05,25.0)),z=(n=5,))))
    for k in (0.5,1.5,3.0,6.0), zi in (1,3,5)
        z = fit.grid.z[zi]
        q = credit_policy(fit,(k=DCGM.discrete_budget(DC_BASE,(k=k,z=z)),z=z))
        @test q.debt ≈ 0 atol=1e-9
        @test q.capital ≈ next_capital(base,(k=k,z=z)) atol=0.02
    end
end

@testset "Discrete credit: collateral, budget and timing" begin
    m = DiscreteCreditModel(DC_BASE;θ=0.5,r=0.02)
    @test leverage(m) ≈ (1+m.r)/(1+m.r-0.5*(1-DC_BASE.depreciation))
    fit = require_converged(solve(m,dc_grid(m)))
    d = fit.diagnostics
    @test d.collateral_slack >= -1e-12
    @test d.budget_error < 1e-12
    @test d.minimum_next_cash >= 0
    @test d.residual_norm < 1e-8
    # Policies on the grid respect the limit and the budget.
    @test all(fit.policy.next_capital .<= leverage(m).*fit.policy.savings .+ 1e-9)
    @test all(fit.policy.consumption .+ fit.policy.savings .≈ coordinates(fit.grid).k)
    path = simulate(fit,(k=2.0,z=fit.grid.z[3]),300;rng=MersenneTwister(7))
    for t in 1:300
        # Capital is bought in t and produces in t+1 at the next productivity draw.
        y = production(DC_BASE,(k=path.capital[t],z=path.states[t+1].z))
        @test path.output[t] ≈ y
        @test path.states[t+1].k ≈ y-(m.r+DC_BASE.depreciation)*path.capital[t]+(1+m.r)*path.savings[t]
        @test path.capital[t] <= leverage(m)*path.savings[t]+1e-9
        @test (1+m.r)*path.debt[t] <= m.θ*(1-DC_BASE.depreciation)*path.capital[t]+1e-9
        @test path.consumption[t]+path.savings[t] ≈ path.states[t].k
    end
    @test any(>(1e-6),path.debt)
    @test all(s -> s.k > 0,path.states)
end

@testset "Discrete credit: capital choice matches brute force" begin
    m = DiscreteCreditModel(DC_BASE;θ=0.5,r=0.02)
    fit = require_converged(solve(m,dc_grid(m;n=100)))
    cache = getfield(fit,:continuation)
    ws = cache.workspace
    for s in (0.1,0.8,2.0,4.0), z in (1,3,5)
        q = DCGM.allocate(m,ws,cache.V,s,z,1;refine=true)
        ks = range(0,ws.λ*s;length=50_001)
        best = maximum(DCGM.expected_value(m,ws,cache.V,s,k,z,1) for k in ks)
        @test q.value >= best-1e-10
        @test 0 <= q.capital <= ws.λ*s
    end
end

@testset "Discrete credit: algorithms and grid refinement agree" begin
    m = DiscreteCreditModel(DC_BASE;θ=0.5,r=0.02)
    g = dc_grid(m;n=80,nz=3)
    mpi = require_converged(solve(m,g))
    vi = require_converged(solve(m,g;algorithm=:value_iteration,maxiters=3000))
    @test maximum(abs,mpi.value.-vi.value) < 1e-6
    coarse = require_converged(solve(m,dc_grid(m;n=150)))
    fine = require_converged(solve(m,dc_grid(m;n=300)))
    for x in (0.8,2.0,4.0,8.0), zi in (2,4)
        a,b = credit_policy(coarse,(k=x,z=coarse.grid.z[zi])),credit_policy(fine,(k=x,z=fine.grid.z[zi]))
        @test a.savings ≈ b.savings atol=0.05
        @test a.capital ≈ b.capital atol=0.1
    end
end

@testset "Discrete credit: deterministic and ability models" begin
    det = DiscreteCreditModel(DiscreteRamseyCassKoopmansModel();θ=0.5,r=0.0)
    fit = require_converged(solve(det,StateGrid(det;k=(n=120,bounds=(0.05,12.0),spacing=:stretched))))
    @test keys(getfield(fit.grid,:axes)) == (:k,)
    @test credit_policy(fit,(k=2.0,)).capital > 0
    types = Population([0.8,1.2],[0.3,0.7])
    base = DiscreteSkibaModel(productivity=LogAR1(persistence=0.9,innovation_std=0.1),population=types)
    m = DiscreteCreditModel(base;θ=0.5,r=0.02)
    fit = require_converged(solve(m,StateGrid(m;k=(n=100,bounds=(0.05,25.0),spacing=:stretched),z=(n=3,))))
    d = stationary(fit)
    @test vec(sum(mass(d);dims=(1,2))) ≈ types.shares
    start = distribution(fit.grid;masses=fill(1/length(fit.grid),size(fit.grid)))
    later = advance(Evolution(fit),start,50)
    @test sum(mass(later)) ≈ 1
    @test vec(sum(mass(later);dims=(1,2))) ≈ [0.5,0.5]
end

@testset "Discrete credit: market clearing" begin
    m = DiscreteCreditModel(DC_BASE;θ=0.5,r=0.02)
    g = dc_grid(m;n=120)
    eq = discrete_credit_equilibrium(DC_BASE,g;θ=0.5)
    r = clearing_rate(eq)
    found = only(eq.equilibria)
    @test -DC_BASE.depreciation < r < 1/DC_BASE.discount-1
    @test abs(found.relative_excess) <= 1e-3
    @test found.borrowing > 0
    @test all(p -> p.excess > 0,filter(p -> p.status == :solved && p.rate < r,collect(eq.scan)))
    @test all(p -> p.excess < 0,filter(p -> p.status == :solved && p.rate > r,collect(eq.scan)))
    # Aggregates recompute from the reported fit and distribution.
    p = vec(mass(found.distribution))
    @test sum(p.*vec(found.fit.policy.next_capital)) ≈ found.capital
    @test sum(p.*vec(found.fit.policy.savings)) ≈ found.own_funds
    # A finite horizon from an explicit start.
    start = distribution(g;masses=fill(1/length(g),size(g)))
    short = discrete_credit_equilibrium(DC_BASE,g;θ=0.5,horizon=50,initial=start,rates=range(0.0,0.035;length=6))
    @test short.settings.long_run == :horizon
    @test length(short.equilibria) == 1 && isfinite(only(short.equilibria).late_change)
    @test_throws ArgumentError discrete_credit_equilibrium(DC_BASE,g;θ=0.5,horizon=50)
    @test_throws ArgumentError discrete_credit_equilibrium(DiscreteSkibaModel(),StateGrid(DiscreteSkibaModel();k=(n=20,bounds=(0.1,5.0)));θ=0.5)
    @test_throws ArgumentError discrete_credit_equilibrium(DC_BASE,g;θ=1.5)
end

@testset "Discrete credit: invalid inputs" begin
    @test_throws ArgumentError DiscreteCreditModel(DC_BASE;θ=-0.1,r=0.0)
    @test_throws ArgumentError DiscreteCreditModel(DC_BASE;θ=1.1,r=0.0)
    @test_throws ArgumentError DiscreteCreditModel(DC_BASE;θ=0.5,r=-DC_BASE.depreciation)
    m = DiscreteCreditModel(DC_BASE;θ=0.5,r=0.02)
    @test_throws ArgumentError solve(m,dc_grid(m;n=20);capital_points=1)
    fit = solve(m,dc_grid(m;n=40);maxiters=0)
    @test !fit.converged
    @test_throws ArgumentError credit_policy(fit,(k=1.0,z=fit.grid.z[1]))
end

@testset "Discrete credit: converges where policy iteration once cycled" begin
    # With golden-section refinement inside the iteration, modified policy iteration
    # alternated between two policies here. On the fixed capital grid the Bellman operator
    # is a contraction and this converges directly; the value-iteration fallback is a
    # safeguard and is reported when used.
    types = Population([0.8,1.2],[0.3,0.7])
    base = DiscreteSkibaModel(productivity=LogAR1(persistence=0.9,innovation_std=0.1),population=types)
    g = StateGrid(DiscreteCreditModel(base;θ=0.5,r=0.0);k=(n=150,bounds=(0.05,25.0),spacing=:stretched),z=(n=7,))
    for r in (0.0265,0.027,0.0275)
        fit = solve(DiscreteCreditModel(base;θ=0.5361065320741775,r=r),g)
        @test fit.converged
        @test fit.diagnostics.residual_norm < 1e-8
        @test fit.diagnostics.switched_to_value_iteration >= 0
    end
end
