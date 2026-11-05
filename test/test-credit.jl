using Test, GrowthModels, ForwardDiff
const CreditGM = GrowthModels

const CREDIT_OU = OrnsteinUhlenbeckProcess(θ=-log(0.9),σ=0.1)
credit_grid(m;n=120,hi=60.0) = StateGrid(m;k=(n=n,bounds=(0.05,hi),spacing=:stretched),z=(n=9,))

@testset "Capital choice matches brute force" begin
    base = StochasticSkibaModel(CREDIT_OU)
    candidates = range(1e-6,40.0;length=400_001)
    for λ in (1.0,1.5,3.0,20.0), r in (-0.03,0.02), a in (0.2,1.0,2.5,6.0), z in (0.8,1.0,1.25)
        m = CreditModel(base;λ=λ,r=r)
        got = capital_choice(m,(k=a,z=z))
        cost = r+base.δ
        profits = [k <= λ*a ? production(base,(k=k,z=z))-cost*k : -Inf for k in candidates]
        @test got.profit ≈ maximum(profits) atol=1e-4
        @test got.capital <= λ*a*(1+eps())
        @test got.profit ≈ production(m,(k=a,z=z))-cost*got.capital atol=1e-12
    end
end

@testset "Adoption threshold falls one for one with λ" begin
    # r+δ=0.1, z=1: low technology is unconstrained at 1.298 and earns 0.3027, and the
    # high technology matches that profit at capital ≈ 3.035.
    base = StochasticSkibaModel(CREDIT_OU)
    threshold(λ) = begin
        m = CreditModel(base;λ=λ,r=0.05)
        lo,hi = 0.01,50.0
        for _ in 1:60
            mid = (lo+hi)/2
            capital_choice(m,(k=mid,z=1.0)).technology == 2 ? (hi=mid) : (lo=mid)
        end
        hi
    end
    @test threshold(1.0) ≈ 3.035 atol=0.01
    for λ in (2.0,4.0,10.0)
        @test threshold(λ) ≈ threshold(1.0)/λ rtol=1e-6
    end
    # Capital jumps at the threshold while profit stays continuous.
    m = CreditModel(base;λ=2.0,r=0.05)
    a = threshold(2.0)
    below,above = capital_choice(m,(k=a-1e-8,z=1.0)),capital_choice(m,(k=a+1e-8,z=1.0))
    @test above.capital-below.capital > 1
    @test above.profit ≈ below.profit atol=1e-6
end

@testset "No borrowing at a low rate reproduces the existing model" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = credit_grid(base)
    autarky = solve(base,g)
    credit = solve(CreditModel(base;λ=1.0,r=-base.δ+1e-7),g)
    require_converged(autarky); require_converged(credit)
    @test maximum(abs,credit.value.-autarky.value) < 1e-9
    @test maximum(abs,credit.policy.consumption.-autarky.policy.consumption) < 1e-9
end

@testset "Budget is profit plus interest less consumption" begin
    base = StochasticSkibaModel(CREDIT_OU)
    m = CreditModel(base;λ=2.0,r=0.02)
    g = credit_grid(m)
    fit = solve(m,g)
    require_converged(fit)
    s = (k=g.k[60],z=g.z[5])
    @test capital_drift(fit,s) ≈ capital_choice(m,s).profit+m.r*s.k-consumption(fit,s) atol=1e-10
end

@testset "Deterministic steady state matches the closed form" begin
    # With r<ρ wealth falls until constrained, where the levered return equals ρ:
    # λ(f′(k)-r-δ)+r=ρ, so f′(k)=r+δ+(ρ-r)/λ and a=k/λ.
    base = RamseyCassKoopmansModel()
    for λ in (1.5,3.0), r in (-0.02,0.02)
        m = CreditModel(base;λ=λ,r=r)
        g = StateGrid(m;k=(n=2000,bounds=(0.01,12.0),spacing=:linear))
        fit = solve(m,g)
        require_converged(fit)
        k = (base.α*base.A/(r+base.δ+(base.ρ-r)/λ))^(1/(1-base.α))
        drift = fit.policy.drift
        crossing = findfirst(i -> drift[i] > 0 && drift[i+1] <= 0,1:length(drift)-1)
        @test g.k[crossing] ≈ k/λ atol=2*(g.k[2]-g.k[1])
        @test capital_choice(m,(k=k/λ,)).capital ≈ k rtol=1e-8
    end
end

@testset "Market clearing" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = credit_grid(base)
    eq = credit_equilibrium(base,g;λ=2.0)
    r = clearing_rate(eq)
    found = only(eq.equilibria)
    @test -base.δ < r < base.ρ
    @test abs(found.capital-found.wealth)/found.wealth <= 1e-3
    @test found.borrowing > 0.05
    @test found.boundary_ok
    # Recomputing from the reported fit gives the reported market aggregates.
    m = CreditModel(base;λ=2.0,r=r)
    d = found.distribution
    @test expectation(d,s -> capital_choice(m,s).capital) ≈ found.capital
    @test expectation(d,s -> s.k) ≈ found.wealth
    # Warm starts save time without deciding the answer.
    cold = stationary(solve(m,g))
    @test maximum(abs,mass(cold).-mass(d)) < 1e-6
    # Excess demand is positive below the rate and negative above it.
    @test all(p -> p.excess > 0,filter(p -> p.rate < r,collect(eq.scan)))
    @test all(p -> p.excess < 0,filter(p -> p.rate > r,collect(eq.scan)))
    # Relaxing the constraint raises the rate.
    @test clearing_rate(credit_equilibrium(base,g;λ=5.0)) > r
    # Reports rather than hides a grid that cannot hold the wealth distribution.
    narrow = credit_grid(base;hi=4.0)
    tight = credit_equilibrium(base,narrow;λ=2.0,rates=range(0.0,0.048;length=4))
    @test any(p -> !p.boundary_ok,tight.scan)
end

@testset "Ability types share one market" begin
    base = StochasticSkibaAbilityModel(CREDIT_OU)
    g = credit_grid(base;n=60)
    eq = credit_equilibrium(base,g;λ=2.0,rates=range(-0.03,0.0475;length=7))
    found = only(eq.equilibria)
    @test abs(found.excess)/found.wealth <= 1e-3
    @test vec(sum(mass(found.distribution);dims=(1,2))) ≈ population(base).shares
end

@testset "ResourceModel can set the population, but not transfers" begin
    types = Population([0.7,1.3],[0.4,0.6])
    wrapped = ResourceModel(StochasticSkibaAbilityModel(CREDIT_OU);population=types)
    m = CreditModel(wrapped;λ=2.0,r=0.02)
    s = (k=1.0,z=1.0,ability=1.3)
    @test production(m,s) ≈ production(wrapped,merge(s,(k=capital_choice(m,s).capital,)))
    @test_throws ArgumentError CreditModel(ResourceModel(StochasticSkibaAbilityModel(CREDIT_OU);transfer=0.1);λ=2.0,r=0.02)
end

@testset "Invalid credit inputs" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = credit_grid(base;n=20)
    @test_throws ArgumentError CreditModel(base;λ=0.5,r=0.0)
    @test_throws ArgumentError CreditModel(base;λ=2.0,r=-base.δ)
    @test_throws ArgumentError CreditModel(StochasticSkibaModel(PoissonProcess(z=[.5,1.],Q=[-.1 .1;.1 -.1]));λ=2.0,r=0.0)
    @test_throws ArgumentError CreditModel(FunctionalModel((k,z,a)->k^.3*z*a,CREDIT_OU,Population([1.],[1.]));λ=2.0,r=0.0)
    @test_throws ArgumentError credit_equilibrium(SkibaModel(),StateGrid(SkibaModel();k=(n=20,bounds=(.1,5.)));λ=2.0)
    @test_throws ArgumentError credit_equilibrium(base,g;λ=1.0)
    @test_throws ArgumentError credit_equilibrium(base,g;λ=ForwardDiff.Dual(2.0,1.0))
    @test_throws ArgumentError credit_equilibrium(base,g;λ=2.0,rates=[0.0,base.ρ])
    @test_throws ArgumentError credit_equilibrium(base,g;λ=2.0,solve_options=(initial=nothing,))
end

@testset "Clearing rate refuses ambiguous results" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = credit_grid(base;n=60)
    none = credit_equilibrium(base,g;λ=2.0,rates=[-0.03,-0.02])
    @test isempty(none.equilibria)
    @test_throws ArgumentError clearing_rate(none)
end

@testset "Excess demand jumps where a productivity state switches technology" begin
    # At seven states z≈0.747 holds 29% of the mass and is indifferent near r=0.03427.
    base = StochasticSkibaModel(CREDIT_OU)
    g = StateGrid(base;k=(n=600,bounds=(0.05,60.0),spacing=:stretched),z=(n=7,))
    z = g.z[2]
    eq = credit_equilibrium(base,g;λ=2.0,rates=[0.034,0.035])
    @test isempty(eq.equilibria)
    jump = only(eq.discontinuities)
    @test jump.reason == :jump
    @test jump.upper_rate-jump.lower_rate <= 1e-5
    @test jump.lower_excess-jump.upper_excess > 0.3
    @test_throws ArgumentError clearing_rate(eq)
    # The bracket sits exactly on that state's technology switch.
    wealthy = (k=1e3,z=z)
    @test capital_choice(CreditModel(base;λ=1e6,r=jump.lower_rate),wealthy).technology == 2
    @test capital_choice(CreditModel(base;λ=1e6,r=jump.upper_rate),wealthy).technology == 1
end

@testset "Transition passes" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = credit_grid(base;n=80)
    fit_autarky = solve(base,g)
    require_converged(fit_autarky)

    # With no borrowing and a constant low rate, the distribution pass is existing Evolution.
    r = -base.δ+1e-7
    m = CreditModel(base;λ=1.0,r=r)
    fit = solve(m,g)
    initial = distribution(g;masses=fill(1/length(g),size(g)))
    cache = CreditGM.CreditPathWorkspace(base,g,1.0,0.5,20,r,Array{Float64}(copy(fit.value)),vec(mass(initial)))
    v = CreditGM.backward_pass!(cache,fill(r,20))
    @test maximum(abs,v.-fit.value) < 1e-6
    CreditGM.forward_pass!(cache)
    expected = mass(advance(Evolution(fit;dt=0.5),initial,10.0))
    @test maximum(abs,reshape(cache.masses[:,end],size(g)).-expected) < 1e-6
    @test maximum(abs,sum(cache.masses;dims=1).-1) < 1e-12

    # Backward induction forgets a wrong terminal value and reaches the stationary solution.
    m2 = CreditModel(base;λ=2.0,r=0.02)
    target = solve(m2,g)
    long = CreditGM.CreditPathWorkspace(base,g,2.0,1.0,1500,0.02,Array{Float64}(copy(fit_autarky.value)),vec(mass(initial)))
    @test maximum(abs,CreditGM.backward_pass!(long,fill(0.02,1500)).-target.value) < 1e-4

    # Time variation matters: a low rate near t=0 only, against the same terminal value.
    rates = fill(0.02,40)
    rates[1:10] .= -0.02
    shifted = CreditGM.CreditPathWorkspace(base,g,2.0,0.5,40,0.02,Array{Float64}(copy(target.value)),vec(mass(initial)))
    @test maximum(abs,CreditGM.backward_pass!(shifted,rates).-target.value) > 1e-3

    # The static rate clears a fixed distribution, and more wealth lowers it.
    states = cache.states
    p = vec(mass(initial))
    A = sum(p[i]*states[i].k for i in eachindex(p))
    rate = CreditGM.static_clearing_rate(base,2.0,states,p,A,0.0)
    @test CreditGM.capital_demand(base,2.0,rate-1e-6,states,p) >= A
    @test CreditGM.capital_demand(base,2.0,rate+1e-6,states,p) <= A
    @test CreditGM.static_clearing_rate(base,2.0,states,p,1.2A,rate) < rate
end

@testset "Transition paths" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = StateGrid(base;k=(n=80,bounds=(0.05,60.0),spacing=:stretched),z=(n=15,))
    eq = credit_equilibrium(base,g;λ=2.0)
    r = clearing_rate(eq)
    found = only(eq.equilibria)

    # Starting at the equilibrium, the economy stays there.
    stay = credit_transition(base,g,found.distribution;λ=2.0,terminal=eq,T=10.0,dt=0.5,
        rate_tolerance=2e-4,maxiters=20)
    require_converged(stay)
    @test maximum(abs,stay.rates.-r) < 1e-3
    @test stay.diagnostics.terminal.total_variation < 1e-3
    @test maximum(abs,stay.value.-found.fit.value) < 1e-3

    autarky = stationary(solve(base,g))
    path = credit_transition(base,g,autarky;λ=2.0,terminal=eq,T=20.0,dt=0.5,rate_tolerance=1e-3,maxiters=100)
    require_converged(path)
    @test path.diagnostics.mass_error < 1e-12
    @test path.wealth[1] ≈ expectation(autarky,s -> s.k)
    @test all(i -> path.excess[i] == path.capital[i]-path.wealth[i],eachindex(path.times))
    @test all(>(0),path.borrowing)
    # Aggregates agree with evaluating the reported rate and distribution directly.
    m = CreditModel(base;λ=2.0,r=path.rates[11])
    d = at(path.distributions,5.0)
    @test expectation(d,s -> capital_choice(m,s).capital) ≈ path.capital[11]
    @test expectation(d,s -> s.k) ≈ path.wealth[11]
    @test expectation(d,s -> production(m,s)) ≈ path.output[11]
    @test expectation(d,s -> float(capital_choice(m,s).technology == 2)) ≈ path.high_technology[11]
    # Credit reallocates capital immediately: output jumps above autarky at t=0.
    @test path.output[1] > expectation(autarky,s -> production(base,s))
    # Results do not depend on the starting guess.
    other = credit_transition(base,g,autarky;λ=2.0,terminal=eq,T=20.0,dt=0.5,rate_tolerance=1e-3,
        maxiters=100,initial_rates=fill(0.02,40))
    @test maximum(abs,other.rates.-path.rates) < 5e-3

    capped = credit_transition(base,g,autarky;λ=2.0,terminal=eq,T=20.0,dt=0.5,maxiters=1,rate_tolerance=1e-12)
    @test !capped.converged
    @test_throws ArgumentError require_converged(capped)
    @test_throws ArgumentError credit_transition(base,g,autarky;λ=3.0,terminal=eq,T=10.0,dt=0.5)
    @test_throws ArgumentError credit_transition(base,g,autarky;λ=2.0,terminal=eq,T=10.0,dt=0.3)
    @test_throws ArgumentError credit_transition(base,g,autarky;λ=2.0,terminal=eq,T=10.0,dt=0.5,damping=0.0)
    @test_throws ArgumentError credit_transition(base,g,autarky;λ=2.0,terminal=eq,T=10.0,dt=0.5,initial_rates=[0.0])
end

@testset "Market clearing at a finite horizon" begin
    base = StochasticSkibaModel(CREDIT_OU)
    g = credit_grid(base)
    start = distribution(g;masses=fill(1/length(g),size(g)))
    stationary_eq = credit_equilibrium(base,g;λ=2.0)
    long = credit_equilibrium(base,g;λ=2.0,horizon=2000.0,initial=start,rates=range(0.03,0.045;length=6))
    short = credit_equilibrium(base,g;λ=2.0,horizon=200.0,initial=start,rates=range(0.025,0.045;length=6))
    @test long.settings.long_run == :horizon && long.settings.horizon == 2000
    @test stationary_eq.settings.long_run == :stationary
    # A long horizon reaches the stationary answer for this well-mixed model. At 200 years
    # from a uniform start the distribution is still moving, and the rate is lower.
    @test clearing_rate(long) ≈ clearing_rate(stationary_eq) atol=2e-4
    @test only(long.equilibria).late_change < 1e-6
    @test only(short.equilibria).late_change > 1e-3
    @test clearing_rate(short) < clearing_rate(stationary_eq)-1e-3
    @test isnan(only(stationary_eq.equilibria).late_change)
    found = only(short.equilibria)
    m = CreditModel(base;λ=2.0,r=found.rate)
    @test expectation(found.distribution,s -> capital_choice(m,s).capital) ≈ found.capital
    @test_throws ArgumentError credit_equilibrium(base,g;λ=2.0,horizon=100.0)
    @test_throws ArgumentError credit_equilibrium(base,g;λ=2.0,horizon=5.0,initial=start)
    @test_throws ArgumentError credit_equilibrium(base,g;λ=2.0,horizon=100.0,initial=start,dt=0.0)
end
