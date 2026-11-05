using Test, GrowthModels, LinearAlgebra, SparseArrays, Random, ForwardDiff

@testset "Native Markov and AR(1) contracts" begin
    values=[0.8,1.2]; probabilities=[0.9 0.1;0.2 0.8]
    p=MarkovChain(values=values,transition=probabilities)
    values[1]=0.1; probabilities[1,1]=0.0
    @test p.values==[0.8,1.2]
    @test transition(p)==[0.9 0.1;0.2 0.8]
    @test_throws ArgumentError (p.transition[1,1]=0.0)
    @test_throws ArgumentError (p.values[1]=0.1)
    for P in ([1.1 -0.1;0.2 0.8],[0.9 0.0;0.2 0.8],[NaN 0;0 1])
        @test_throws ArgumentError MarkovChain(values=[0.8,1.2],transition=P)
    end
    @test_throws DimensionMismatch MarkovChain(values=[1.0],transition=Matrix{Float64}(I,2,2))
    @test_throws ArgumentError MarkovChain(values=[0.0],transition=ones(1,1))
    @test_throws ArgumentError LogAR1(persistence=1.0)
    @test_throws ArgumentError LogAR1(innovation_std=-0.1)
    @test_throws ArgumentError ForwardDiff.derivative(x->LogAR1(innovation_std=x).innovation_std,0.1)
    for rho in (-0.5,0.0,0.9,0.99), n in (2,7,11)
        ar=LogAR1(persistence=rho,innovation_std=0.08,log_mean=-0.2)
        m=DiscreteRamseyCassKoopmansModel(productivity=ar)
        g=StateGrid(m;k=[0.01,5.0],z=(n=n,))
        P=transition(ar,g.z)
        x=log.(g.z)
        pi=[binomial(n-1,j)/2.0^(n-1) for j in 0:n-1]
        @test P'*pi ≈ pi atol=1e-13
        @test P*x ≈ -0.2 .+ rho.*(x .+ 0.2) atol=1e-12
        @test P*(x.^2) .- (P*x).^2 ≈ fill(0.08^2,n) atol=1e-12
        @test dot(pi,x) ≈ -0.2 atol=1e-13
        @test dot(pi,(x .+ 0.2).^2) ≈ 0.08^2/(1-rho^2) rtol=1e-12
        @test all(P .>= 0)
        @test vec(sum(P;dims=2)) ≈ ones(n) atol=1e-14
        @test g.kinds.z == :categorical
    end
    ar=LogAR1(innovation_std=0.0,log_mean=log(1.3))
    m=DiscreteSkibaModel(productivity=ar)
    g=StateGrid(m;k=[0.01,8.0],z=(n=7,))
    @test length(g.z)==1
    @test g.z[1]≈1.3
    @test transition(ar,g.z)==ones(1,1)
    @test_throws ArgumentError StateGrid(DiscreteSkibaModel(productivity=LogAR1());z=(n=1,))
    @test_throws ArgumentError StateGrid(m;z=(n=3,width=4))
    @test_throws ArgumentError transition(p,[0.7,1.2])
end

@testset "Ability, timing and deterministic limits" begin
    support=collect(range(0.01,10.0;length=65))
    for ctor in (DiscreteSkibaModel,DiscreteRamseyCassKoopmansModel)
        deterministic=ctor()
        basefit=solve(deterministic,StateGrid(deterministic;k=support))
        for shock in (MarkovChain(values=[1.0],transition=ones(1,1)),LogAR1(innovation_std=0.0))
            m=ctor(productivity=shock)
            fit=solve(m,StateGrid(m;k=support))
            @test fit.converged
            @test vec(fit.value) ≈ basefit.value atol=1e-9
            @test vec(fit.policy.next_capital) ≈ basefit.policy.next_capital atol=1e-9
            @test next_capital(fit,(k=2.3,z=1.0)) ≈ next_capital(basefit,(k=2.3,)) atol=1e-10
        end
        ability=Population([0.8,1.2],[0.3,0.7])
        m=ctor(population=ability)
        g=StateGrid(m;k=support)
        @test state_schema(m)==(:k,:ability)
        @test size(g)==(65,2)
        # Separate solves can stop an iteration apart, so compare them more tightly
        # than these assertions rather than at the default stopping tolerance.
        tight=(value_tolerance=1e-10,residual_tolerance=1e-10)
        fit=solve(m,g;tight...)
        @test fit.converged
        for (a,type) in enumerate(ability.values)
            isolated=ctor(population=Population([type],[1.0]))
            single=solve(isolated,StateGrid(isolated;k=support);tight...)
            @test fit.value[:,a] ≈ vec(single.value) atol=1e-8
            @test fit.policy.next_capital[:,a] ≈ vec(single.policy.next_capital) atol=1e-8
        end
        other=ctor(population=Population([0.8,1.2],[0.9,0.1]))
        @test solve(other,g;tight...).value==fit.value
        frozen=ctor(productivity=MarkovChain(values=[0.8,1.2],transition=Matrix{Float64}(I,2,2)))
        frozenfit=solve(frozen,StateGrid(frozen;k=support);tight...)
        @test frozenfit.value≈fit.value atol=1e-8
        @test frozenfit.policy.next_capital≈fit.policy.next_capital atol=1e-8
        @test_throws ArgumentError next_capital(fit,(k=2.0,ability=1.0))
        @test_throws ArgumentError next_capital(fit,(k=2.0,))
        @test_throws ArgumentError StateGrid(m;ability=[0.9,1.1])
        @test_throws ArgumentError ctor(population=Population([0.0],[1.0]))
        @test_throws ArgumentError ctor(productivity=OrnsteinUhlenbeckProcess(θ=0.1,σ=0.1))
    end
    m=DiscreteSkibaModel(productivity=MarkovChain(values=[0.8,1.2],transition=[0.9 0.1;0.2 0.8]),
        population=Population([0.7,1.3],[0.4,0.6]))
    g=StateGrid(m;k=(n=50,bounds=(0.01,12.0)))
    @test technology_crossing(m) in g.k
    @test_throws ArgumentError production(m,(k=2.0,))
    @test_throws ArgumentError production(m,(k=2.0,z=0.8))
    @test production(m,(k=2.0,z=1.2,ability=1.3)) ≈ 1.2*1.3*production(DiscreteSkibaModel(),(k=2.0,))
    wrapped=ResourceModel(ResourceModel(m;transfer=0.1,income_floor=0.8);transfer=0.05)
    wg=StateGrid(wrapped;k=(n=50,bounds=(0.01,12.0)))
    @test state_schema(wrapped)==(:k,:z,:ability)
    @test length(wg.k)>length(g.k)
    @test solve(wrapped,wg).converged
    abilitywrap=ResourceModel(DiscreteSkibaModel(population=population(m));income_floor=0.5)
    @test size(StateGrid(abilitywrap;k=support))==(65,2)
    @test solve(abilitywrap,StateGrid(abilitywrap;k=support)).converged
    @test last(StateGrid(m;k=(n=20,)).k)>last(StateGrid(DiscreteSkibaModel();k=(n=20,)).k)
end

@testset "Joint Bellman operator and off-grid policy" begin
    shock=MarkovChain(values=[0.8,1.25],transition=[0.85 0.15;0.25 0.75])
    m=DiscreteSkibaModel(productivity=shock,population=Population([0.8,1.2],[0.3,0.7]))
    g=StateGrid(m;k=(n=55,bounds=(0.01,12.0)))
    fit=solve(m,g)
    @test fit.converged
    @test size(fit.value)==size(g)==size(fit.policy.consumption)
    @test fit.diagnostics.budget_error<1e-14
    @test fit.diagnostics.admissible
    reference=solve(m,g;algorithm=:value_iteration)
    @test reference.converged
    @test maximum(abs.(fit.value-reference.value))<3e-7
    @test maximum(abs.(fit.policy.next_capital-reference.policy.next_capital))<1e-7
    P=copy(fit.transition)
    @test all(nonzeros(P).>=0)
    @test maximum(abs.(sum(P;dims=2).-1))<1e-14
    ws=GrowthModels.BellmanWorkspace(m,g)
    ws.capital.=vec(fit.policy.next_capital)
    arbitrary=reshape(sin.(collect(1.0:length(g))),size(g))
    @test GrowthModels.policy_action(ws,g.k,vec(arbitrary)) ≈ P*vec(arbitrary) atol=1e-13
    reward=GrowthModels.discrete_utility.(vec(fit.policy.consumption),m.risk_aversion)
    @test maximum(abs.(reward+m.discount*P*vec(fit.value)-vec(fit.value)))≈fit.diagnostics.residual_norm atol=1e-12
    @test maximum(abs.((I-m.discount*P)\reward-vec(fit.value)))<1e-7
    # Independently assemble expectation and densely enumerate capital choices.
    for (zi,z) in enumerate(g.z), (ai,a) in enumerate(g.ability), k in (0.4,2.3,4.1)
        state=(k=k,z=z,ability=a)
        B=resources(m,state)+(1-m.depreciation)*k
        W=copy(fit.value[:,:,ai])*copy(shock.transition[zi,:])
        actions=range(first(g.k),min(B,last(g.k));length=10001)
        objectives=map(actions) do kp
            j=clamp(searchsortedlast(g.k,kp),1,length(g.k)-1)
            w=(kp-g.k[j])/(g.k[j+1]-g.k[j])
            GrowthModels.discrete_utility(B-kp,m.risk_aversion)+m.discount*((1-w)*W[j]+w*W[j+1])
        end
        kp=next_capital(fit,state); c=consumption(fit,state)
        j=clamp(searchsortedlast(g.k,kp),1,length(g.k)-1)
        w=(kp-g.k[j])/(g.k[j+1]-g.k[j])
        q=GrowthModels.discrete_utility(c,m.risk_aversion)+m.discount*((1-w)*W[j]+w*W[j+1])
        @test q>=maximum(objectives)-1e-10
        @test c+kp≈B
    end
    @test_throws ArgumentError next_capital(fit,(k=2.0,z=1.0,ability=0.8))
    @test_throws ArgumentError fit.continuation
    @test :continuation ∉ propertynames(fit)
    @test_throws ArgumentError (fit.policy.next_capital[1,1,1]=1.0)
    @test_throws ArgumentError (fit.diagnostics.choice_gap[1,1,1]=0.0)
    warm=solve(m,g;initial=fit)
    @test warm.converged
    @test warm.initial_value==fit.value
    changed=DiscreteSkibaModel(productivity=MarkovChain(values=shock.values,transition=[0.6 0.4;0.1 0.9]),
        population=population(m))
    changedwarm=solve(changed,g;initial=fit)
    changedcold=solve(changed,g)
    @test changedwarm.value≈changedcold.value atol=1e-7
    @test changedwarm.policy.next_capital≈changedcold.policy.next_capital atol=1e-7
    fine=StateGrid(m;k=(n=70,bounds=(0.02,11.0)))
    @test solve(m,fine;initial=regrid(fit,fine)).converged
    other=DiscreteSkibaModel(productivity=MarkovChain(values=[0.9,1.3],transition=transition(shock)),population=population(m))
    @test_throws ArgumentError regrid(fit,StateGrid(other;k=g.k))
    @test_throws ArgumentError solve(other,g)
end

@testset "Integer evolution, ability shares and simulation" begin
    m=DiscreteRamseyCassKoopmansModel(discount=0.9,depreciation=1.0,risk_aversion=1.0,
        productivity=MarkovChain(values=[0.8,1.2],transition=[0.9 0.1;0.2 0.8]),
        population=Population([0.8,1.2],[0.3,0.7]))
    g=StateGrid(m;k=(n=90,bounds=(0.001,3.0)))
    fit=solve(m,g)
    @test fit.converged
    # Log/full-depreciation policy remains alpha*beta*current output under risk.
    for z in g.z, a in g.ability, k in (0.1,0.5,1.5)
        s=(k=k,z=z,ability=a)
        @test next_capital(fit,s)≈m.α*m.discount*production(m,s) atol=0.025
    end
    initial=distribution(g;masses=fill(1/length(g),size(g)))
    e=Evolution(fit)
    @test e isa DiscreteEvolution
    @test e isa Evolution
    @test_throws ArgumentError Evolution(fit;dt=1)
    @test_throws ArgumentError advance(e,initial,1.0)
    @test_throws ArgumentError trajectory(e,initial,[0,0.5,1])
    @test_throws ArgumentError trajectory(e,initial,[1,0])
    path=trajectory(e,initial,[0,2,5,10])
    @test mass(path,0)==mass(initial)
    @test mass(path,10)≈reshape(copy(fit.transition)'^10*vec(mass(initial)),size(g)) atol=1e-14
    @test mass(advance(e,advance(e,initial,3),7))≈mass(path,10) atol=1e-14
    for t in path.times
        @test sum(mass(path,t))≈1 atol=1e-13
        @test vec(sum(mass(path,t);dims=(1,2)))≈[0.5,0.5] atol=1e-13
    end
    shared=trajectory(e,initial,[2,3,5,7,10])
    @test mass(shared,5)==mass(path,5)
    @test mass(final(path))==mass(path,10)
    observations=observe(e,initial,[0,2,5,10],(capital=s->s.k,ability=s->s.ability))
    @test observations.values.capital≈expectation(path,s->s.k)
    @test observations.values.ability≈ones(4)
    @test_throws ArgumentError (path.probabilities[1,1]=0)
    steady=stationary(fit;require_aperiodic=true)
    @test vec(sum(mass(steady);dims=(1,2)))≈[0.3,0.7] atol=1e-12
    @test mass(advance(e,steady,5))≈mass(steady) atol=1e-12
    chosen=stationary(fit,initial)
    @test vec(sum(mass(chosen);dims=(1,2)))≈[0.5,0.5] atol=1e-12
    @test mass(advance(e,initial,300))≈mass(chosen) atol=1e-9
    initialstate=(k=0.5,z=0.8,ability=1.2)
    sim=simulate(fit,initialstate,30;rng=MersenneTwister(13))
    @test sim==simulate(fit,initialstate,30;rng=MersenneTwister(13))
    @test length(sim.states)==31
    @test sim.times==collect(0:30)
    @test all(s->s.ability==1.2,sim.states)
    for t in 1:30
        @test sim.states[t+1].k==next_capital(fit,sim.states[t])
        @test sim.consumption[t]+sim.states[t+1].k≈production(m,sim.states[t])
    end
    rng=MersenneTwister(913)
    high=count(1:2000) do _
        simulate(fit,initialstate,1;rng).states[end].z==1.2
    end
    @test abs(high/2000-0.1)<6sqrt(0.1*0.9/2000)
    @test length(simulate(fit,initialstate,0).states)==1
    @test_throws ArgumentError simulate(fit,initialstate,-1)
    # Ability-only evolution uses the same block convention without a z axis.
    abilityonly=DiscreteRamseyCassKoopmansModel(depreciation=1.0,risk_aversion=1.0,population=population(m))
    ag=StateGrid(abilityonly;k=g.k)
    af=solve(abilityonly,ag)
    @test vec(sum(mass(stationary(af));dims=1))≈[0.3,0.7]
    zeroshare=DiscreteRamseyCassKoopmansModel(depreciation=1.0,risk_aversion=1.0,
        population=Population([0.8,1.2],[0.0,1.0]))
    @test vec(sum(mass(stationary(solve(zeroshare,ag)));dims=1))≈[0.0,1.0]
end

@testset "Closed classes, absorption and periodicity" begin
    P=sparse([0.0 0.25 0.75;0 1 0;0 0 1])
    @test_throws ArgumentError GrowthModels.stationary_transition(P)
    @test GrowthModels.stationary_transition(P;initial_mass=[1.0,0,0])≈[0,0.25,0.75]
    cycle=sparse([0.0 1;1 0])
    @test GrowthModels.stationary_transition(cycle)≈[0.5,0.5]
    @test GrowthModels.stationary_transition(cycle;initial_mass=[1.0,0])≈[0.5,0.5]
    @test_throws ArgumentError GrowthModels.stationary_transition(cycle;require_aperiodic=true)
    # A zero-mass periodic class should not prevent an aperiodic mixture.
    joint=sparse([1.0 0 0;0 0 1;0 1 0])
    @test GrowthModels.stationary_transition(joint;initial_mass=[1.0,0,0],require_aperiodic=true)==[1,0,0]
    @test_throws ArgumentError GrowthModels.stationary_transition(joint;initial_mass=[0.0,1,0],require_aperiodic=true)
    # Test the public method with a genuine model whose productivity alternates.
    m=DiscreteRamseyCassKoopmansModel(depreciation=1.0,risk_aversion=1.0,
        productivity=MarkovChain(values=[0.9,1.1],transition=Matrix(cycle)))
    g=StateGrid(m;k=(n=50,bounds=(0.01,2.0)))
    fit=solve(m,g)
    initial=distribution(g;masses=fill(1/length(g),size(g)))
    invariant=stationary(fit,initial)
    @test mass(advance(Evolution(fit),invariant,1))≈mass(invariant) atol=1e-12
    @test_throws ArgumentError stationary(fit,initial;require_aperiodic=true)
end

@testset "Invariant probabilities tolerate solver roundoff" begin
    # Tail probabilities of a fine shock chain are near machine precision, so a
    # direct solve returns tiny negative entries for these valid chains.
    for (m,nk,nz,upper) in ((DiscreteRamseyCassKoopmansModel(A=1.0,productivity=LogAR1()),200,11,6.0),
                            (DiscreteSkibaModel(productivity=LogAR1()),300,21,20.0))
        g=StateGrid(m;k=(n=nk,bounds=(0.2,upper)),z=(n=nz,))
        fit=solve(m,g)
        @test fit.converged
        initial=distribution(g;masses=fill(1/length(g),size(g)))
        invariant=stationary(fit,initial)
        @test all(>=(0),mass(invariant))
        @test sum(mass(invariant))≈1 atol=1e-13
        @test mass(advance(Evolution(fit),invariant,1))≈mass(invariant) atol=1e-12
    end
    # Roundoff is removed; a materially negative solve remains an error.
    cleaned=GrowthModels.clip_probability_roundoff!([0.5,-1e-16,0.5])
    @test cleaned[2]==0
    @test sum(cleaned)≈1 atol=1e-15
    @test_throws ArgumentError GrowthModels.clip_probability_roundoff!([0.6,-0.1,0.5])
end

@testset "Policy jumps with shocks and types" begin
    shock=MarkovChain(values=[0.8,1.25],transition=[0.85 0.15;0.25 0.75])
    types=Population([0.8,1.2],[0.5,0.5])
    m=DiscreteSkibaModel(productivity=shock,population=types)
    fit=solve(m,StateGrid(m;k=(n=1000,bounds=(0.01,12.0))))
    @test_throws ArgumentError policy_jumps(fit)
    @test_throws ArgumentError policy_jumps(fit;z=1.25)
    @test_throws ArgumentError policy_jumps(fit;ability=1.2)
    @test_throws ArgumentError policy_jumps(fit;z=1.0,ability=1.2)
    @test_throws ArgumentError policy_jumps(fit;z=1.25,ability=1.0)
    jumps=policy_jumps(fit;z=1.25,ability=1.2)
    @test !isempty(jumps)
    state(x)=(k=x,z=1.25,ability=1.2)
    for j in jumps
        @test next_capital(fit,state(j.k-1e-9)) ≈ j.next_capital_below atol=1e-7
        @test next_capital(fit,state(j.k+1e-9)) ≈ j.next_capital_above atol=1e-7
    end
    # Slices are independent: each uses its own shock and type.
    @test policy_jumps(fit;z=0.8,ability=0.8)!=jumps
    # The largest jump persists under refinement, though the count need not.
    finer=solve(m,StateGrid(m;k=(n=2000,bounds=(0.01,12.0))))
    largest(list)=list[argmax([j.jump for j in list])]
    @test largest(policy_jumps(finer;z=1.25,ability=1.2)).k ≈ largest(jumps).k atol=0.01
    @test largest(policy_jumps(finer;z=1.25,ability=1.2)).jump ≈ largest(jumps).jump atol=0.005
    # Axes that are absent cannot be supplied, and present ones are required.
    abilityonly=DiscreteSkibaModel(population=types)
    af=solve(abilityonly,StateGrid(abilityonly;k=(n=300,bounds=(0.01,12.0))))
    @test policy_jumps(af;ability=1.2) isa Vector
    @test_throws ArgumentError policy_jumps(af;z=1.0,ability=1.2)
    @test_throws ArgumentError policy_jumps(af)
end
