using Test, GrowthModels, LinearAlgebra, SparseArrays, ForwardDiff

@testset "Discrete economics and numerical domain" begin
    for ctor in (DiscreteRamseyCassKoopmansModel,DiscreteSkibaModel)
        for kw in ((discount=1.0,),(discount=0.0,),(depreciation=-0.1,),
                   (depreciation=1.1,),(risk_aversion=0.0,),(α=1.0,))
            @test_throws ArgumentError ctor(;kw...)
        end
        m=ctor()
        @test preferences(m).discount == 0.96
        @test process(m) === nothing
        @test production(m,(k=2.0,)) > 0
        @test_throws ArgumentError StateGrid(m;k=[-1.0,1.0])
        @test_throws ArgumentError StateGrid(m;k=[1.0,1.0])
        @test_throws ArgumentError StateGrid(m;k=(n=10,bound=(0.1,2.0)))
        @test_throws ArgumentError StateGrid(m;z=[1.0])
        @test_throws ArgumentError solve(m,StateGrid(m);algorithm=:local)
    end
    m=DiscreteSkibaModel()
    crossing=technology_crossing(m)
    g=StateGrid(m;k=(n=40,bounds=(0.01,20.0)))
    @test crossing in g.k
    explicit=StateGrid(m;k=[0.01,20.0])
    @test collect(explicit.k)==[0.01,20.0]
    wrapped=ResourceModel(m;transfer=0.1,income_floor=0.8)
    wg=StateGrid(wrapped;k=(n=40,bounds=(0.01,20.0)))
    @test length(wg)>length(g)
    @test resources(wrapped,(k=0.1,))==0.8
    @test solve(wrapped,wg).converged
    nested=ResourceModel(wrapped;transfer=0.05,income_floor=0.9)
    ng=StateGrid(nested;k=(n=40,bounds=(0.01,20.0)))
    @test solve(nested,ng).converged
    @test resources(nested,(k=0.1,))==0.9
    @test_throws ArgumentError ForwardDiff.derivative(x->DiscreteSkibaModel(discount=x).discount,0.96)
    @test_throws ArgumentError ForwardDiff.derivative(x->StateGrid(m;k=[x,20.0]).k[1],0.01)
    @test_throws ArgumentError solve(m,StateGrid(m;k=[0.0,1.0,2.0]))
    finite=DiscreteRamseyCassKoopmansModel(risk_aversion=0.5)
    fit=solve(finite,StateGrid(finite;k=(n=40,bounds=(0.0,4.0))))
    @test fit.converged
    @test fit.value[1]==0
    @test next_capital(fit,(k=0.0,))==0
    @test consumption(fit,(k=0.0,))==0
    @test_throws ArgumentError solve(m,StateGrid(m;k=[100.0,101.0]))
end

# Independent dense action search: interpolate with an explicit bracket loop,
# rather than using the Bellman candidate or interpolation implementation.
function dense_reference(B,k,v,β,γ;n=20001)
    actions=range(first(k),min(B,last(k));length=n)
    vals=map(actions) do x
        j=findfirst(i -> k[i] <= x <= k[i+1],1:length(k)-1)
        w=(x-k[j])/(k[j+1]-k[j])
        c=B-x
        u=c==0 ? (γ<1 ? 0.0 : -Inf) : γ==1 ? log(c) : c^(1-γ)/(1-γ)
        u+β*((1-w)*v[j]+w*v[j+1])
    end
    q,i=findmax(vals)
    return q,actions[i]
end

@testset "Global Bellman search" begin
    k=[0.0,0.2,0.8,1.0,1.7,2.0,3.0]
    # Increasing but strongly nonconcave continuation, plus decreasing and flat cases.
    for v in ([0.0,0.8,0.9,1.0,4.0,4.1,4.2],reverse(collect(1.0:7.0)),zeros(7)),
            γ in (0.5,1.0,2.0), B in (0.15,1.3,3.0,5.0)
        β=0.95
        q=GrowthModels.bellman_choice(B,k,v,diff(v)./diff(k),β,γ;alternatives=true)
        dense,action=dense_reference(B,k,v,β,γ)
        @test q.value >= dense-1e-12
        @test q.value-dense < 0.001
        @test q.consumption+q.capital ≈ B
        @test 0 <= q.capital <= min(B,last(k))
    end
    # Exact equality between the two endpoint maxima of a convex objective:
    # log(2-x) + linear continuation has a concave segment, so construct a
    # downward then upward continuation whose endpoint rewards agree.
    k=[0.0,0.5,1.0]; v=[0.0,-10.0,log(2)/0.9]
    q=GrowthModels.bellman_choice(2.0,k,v,diff(v)./diff(k),0.9,1.0;alternatives=true)
    @test q.capital==0.0
    @test q.alternative_capital==1.0
    @test q.choice_gap==0.0
end

@testset "Analytical log growth and solver agreement" begin
    m=DiscreteRamseyCassKoopmansModel(discount=0.9,depreciation=1.0,
        risk_aversion=1.0,A=1.0,α=0.3)
    g=StateGrid(m;k=(n=500,bounds=(0.001,2.0),spacing=:stretched,power=1.5))
    fit=solve(m,g;history=true)
    @test fit.converged
    @test fit.diagnostics.admissible
    @test fit.diagnostics.budget_error < 1e-14
    @test fit.diagnostics.residual_norm < 1e-8
    β,α=m.discount,m.α
    slope=α/(1-α*β)
    intercept=(log(1-α*β)+α*β/(1-α*β)*log(α*β))/(1-β)
    @test maximum(abs.(fit.value .- (intercept .+ slope.*log.(g.k)))) < 0.003
    for x in (0.02,0.12,0.45,0.9,1.8)
        kp=next_capital(fit,(k=x,))
        @test kp ≈ α*β*x^α atol=0.003
        @test consumption(fit,(k=x,))+kp ≈ x^α
    end
    small=StateGrid(m;k=(n=80,bounds=(0.001,2.0)))
    reference=solve(m,small;algorithm=:value_iteration)
    accelerated=solve(m,small)
    @test reference.converged && accelerated.converged
    @test maximum(abs.(reference.value.-accelerated.value)) < 1e-7
    @test maximum(abs.(reference.policy.next_capital.-accelerated.policy.next_capital)) < 1e-7
    @test accelerated.diagnostics.iterations < reference.diagnostics.iterations
    P=copy(fit.transition)
    @test minimum(nonzeros(P)) >= 0
    @test maximum(abs.(sum(P;dims=2).-1)) < 1e-14
    reward=log.(fit.policy.consumption)
    @test maximum(abs.(reward+β*P*fit.value-fit.value)) ≈ fit.diagnostics.residual_norm atol=1e-12
    # Fixed-policy value agrees with the converged Bellman value.
    @test maximum(abs.((I-β*P)\reward-fit.value)) < 1e-7
end

@testset "Discrete ownership, failures, warm starts, queries" begin
    m=DiscreteSkibaModel()
    g=StateGrid(m;k=(n=80,bounds=(0.01,12.0)))
    fit=solve(m,g)
    @test fit.converged
    @test_throws ArgumentError (fit.value[1]=0)
    @test_throws ArgumentError (fit.policy.next_capital[1]=0)
    @test_throws ArgumentError (fit.transition[1,1]=0)
    @test_throws ArgumentError (fit.diagnostics.choice_gap[1]=0)
    @test_throws ArgumentError next_capital(fit,(k=100.0,))
    @test_throws ArgumentError consumption(fit,(z=1.0,))
    @test_throws ArgumentError solve(m,g;initial=fill(NaN,size(g)))
    @test_throws DimensionMismatch solve(m,g;initial=zeros(2))
    @test_throws ArgumentError solve(m,g;maxiters=-1)
    failed=solve(m,g;maxiters=0)
    @test_throws ArgumentError solve(m,g;initial=[isodd(i) ? 1e308 : -1e308 for i in 1:length(g)])
    limited=solve(m,g;maxiters=1)
    @test !limited.converged
    @test limited.diagnostics.stop_reason==:iteration_limit
    @test limited.diagnostics.admissible
    bellman_values=[GrowthModels.bellman_choice(GrowthModels.discrete_budget(m,k),g.k,
        limited.value,diff(limited.value)./diff(g.k),m.discount,m.risk_aversion).value for k in g.k]
    @test maximum(abs.(bellman_values.-limited.value))==limited.diagnostics.residual_norm
    initial=zeros(length(g))
    snapshot=solve(m,g;initial,maxiters=0)
    initial[1]=123.0
    @test snapshot.initial_value[1]==0
    @test !failed.converged
    @test all(isfinite,failed.value)
    @test_throws ArgumentError next_capital(failed,(k=2.0,))
    @test_throws ArgumentError require_converged(failed)
    warm=solve(m,g;initial=fit)
    @test warm.converged
    @test warm.diagnostics.iterations < fit.diagnostics.iterations
    @test warm.initial_value==fit.value
    narrower=StateGrid(m;k=(n=100,bounds=(0.02,10.0)))
    @test_throws ArgumentError solve(m,narrower;initial=fit)
    guess=regrid(fit,narrower)
    @test solve(m,narrower;initial=guess).converged
    wider=StateGrid(m;k=[0.001,20.0])
    @test_throws ArgumentError regrid(fit,wider)
    @test all(isfinite,regrid(fit,wider;extrapolate=true).value)
    # Off-grid policy independently matches dense action maximization.
    for x in (0.4,2.3,technology_crossing(m),8.3)
        B=resources(m,(k=x,))+(1-m.depreciation)*x
        dense,_=dense_reference(B,collect(g.k),copy(fit.value),m.discount,m.risk_aversion)
        kp=next_capital(fit,(k=x,))
        c=consumption(fit,(k=x,))
        q=GrowthModels.discrete_utility(c,m.risk_aversion)+m.discount*
            GrowthModels.interpolate_grid(fit.value,g,(k=kp,))
        @test q >= dense-1e-10
        @test c+kp ≈ B
    end
end

@testset "Skiba threshold and path refinement" begin
    m=DiscreteSkibaModel()
    user_cost=1/m.discount-1+m.depreciation
    low_state=(m.α*m.A_L/user_cost)^(1/(1-m.α))
    high_state=(m.α*m.A_H/user_cost)^(1/(1-m.α))+m.κ
    function threshold(fit)
        lo,hi=2.8,3.5
        @test next_capital(fit,(k=lo,)) < lo
        @test next_capital(fit,(k=hi,)) > hi
        for _ in 1:40
            mid=(lo+hi)/2
            next_capital(fit,(k=mid,)) > mid ? (hi=mid) : (lo=mid)
        end
        (lo+hi)/2
    end
    fits=[solve(m,StateGrid(m;k=(n=n,bounds=(0.001,20.0)))) for n in (500,1000)]
    @test all(f->f.converged,fits)
    thresholds=threshold.(fits)
    @test abs(thresholds[1]-thresholds[2]) < 0.002
    @test abs(thresholds[2]-technology_crossing(m)) > 0.1
    for (fit,t) in zip(fits,thresholds)
        left=next_capital(fit,(k=t-1e-6,))
        right=next_capital(fit,(k=t+1e-6,))
        @test right-left > 0.05
        @test left < t < right
        @test !any(fit.diagnostics.boundary.lower_choice)
        @test !any(fit.diagnostics.boundary.upper_choice)
        spacing=fit.grid.k[3]-fit.grid.k[2]
        for (start,high) in ((t-0.05,false),(t+0.05,true))
            k=start
            for _ in 1:200
                kp=next_capital(fit,(k=k,))
                @test consumption(fit,(k=k,))+kp ≈ resources(m,(k=k,))+(1-m.depreciation)*k
                k=kp
            end
            # Both branch steady states solve 1 = β(f′(k) + 1 - δ). Piecewise-linear
            # continuation pins paths to a nearby node, so compare at grid resolution.
            @test abs(k-(high ? high_state : low_state)) < spacing
        end
        # Distinct local maxima persist near the investment switch.
        q=GrowthModels.bellman_choice(GrowthModels.discrete_budget(m,t),fit.grid.k,
            fit.value,diff(fit.value)./diff(fit.grid.k),m.discount,m.risk_aversion;alternatives=true)
        @test isfinite(q.alternative_capital)
        @test abs(q.capital-q.alternative_capital) > 0.05
        @test q.choice_gap < 1e-8
    end
end

@testset "Discrete numerical breakdown retains finite iterate" begin
    m=DiscreteRamseyCassKoopmansModel()
    # Utility is finite, but its variation over this extreme interval cannot
    # form finite Float64 continuation slopes after the first improvement.
    g=StateGrid(m;k=[1e-300,2e-300])
    fit=solve(m,g;maxiters=2)
    @test !fit.converged
    @test fit.diagnostics.stop_reason==:nonfinite
    @test all(isfinite,fit.value)
    @test fit.value==fit.initial_value
    @test isfinite(fit.diagnostics.residual_norm)
    @test fit.diagnostics.admissible
    @test all(isfinite,fit.policy.consumption)
end

@testset "Smooth steady state from both sides" begin
    m=DiscreteRamseyCassKoopmansModel()
    state=(m.α*m.A/(1/m.discount-1+m.depreciation))^(1/(1-m.α))
    g=StateGrid(m;k=(n=500,bounds=(0.05*state,1.5*state)))
    fit=solve(m,g)
    @test fit.converged
    spacing=g.k[2]-g.k[1]
    for start in (0.5*state,1.4*state)
        k=start
        for _ in 1:500
            k=next_capital(fit,(k=k,))
        end
        @test abs(k-state) < spacing
    end
end

# The per-period economy with period Δ has budget c + k′ = Δf(k) + (1-δΔ)k and
# discount exp(-ρΔ). CRRA homogeneity makes its policy independent of the constant
# scale between period consumption and flow reward, so c/Δ approaches the HJB
# policy. This mapping is a validation device only: the package performs no conversion.
@testset "Continuous-time limit of the Skiba economy" begin
    ρ,δ=0.05,0.05
    hjb=SkibaModel(ρ=ρ,δ=δ)
    hjbgrid=StateGrid(hjb;k=(n=2000,bounds=(0.01,8.0)))
    hjbfit=solve(hjb,hjbgrid;maxiters=5000)
    @test hjbfit.converged
    drift=copy(hjbfit.policy.drift)
    # Above the low steady state, the only negative-to-positive drift switch.
    switches=[hjbgrid.k[i] for i in 1:length(drift)-1 if hjbgrid.k[i]>2 && drift[i]<0 && drift[i+1]>0]
    @test length(switches)==1
    target=only(switches)
    errors=map(((1.0,1000),(0.5,2000))) do (Δ,n)
        m=DiscreteSkibaModel(discount=exp(-ρ*Δ),depreciation=δ*Δ,risk_aversion=hjb.γ,
            A_H=hjb.A_H*Δ,A_L=hjb.A_L*Δ,α=hjb.α,κ=hjb.κ)
        fit=solve(m,StateGrid(m;k=(n=n,bounds=(0.01,8.0))))
        @test fit.converged
        lo,hi=2.0,2.8
        @test next_capital(fit,(k=lo,))<lo
        @test next_capital(fit,(k=hi,))>hi
        for _ in 1:40
            mid=(lo+hi)/2
            next_capital(fit,(k=mid,))>mid ? (hi=mid) : (lo=mid)
        end
        for x in (1.5,2.0,4.0,5.0)
            @test consumption(fit,(k=x,))/Δ ≈ consumption(hjbfit,(k=x,)) atol=0.01
        end
        (lo+hi)/2-target
    end
    # Same-signed first-order error: halving the period halves the threshold gap.
    @test all(>(0),errors)
    @test 1.5 < errors[1]/errors[2] < 2.5
    @test errors[2] < 0.06
end

@testset "Preferences share names across time domains" begin
    native=DiscreteSkibaModel(discount=0.9,depreciation=0.1,risk_aversion=3.0)
    @test preferences(native)==(γ=3.0,discount=0.9,δ=0.1)
    @test preferences(ResourceModel(native;transfer=0.1))==preferences(native)
    # Generic code reads risk aversion and depreciation without knowing the domain.
    shared(m)=(preferences(m).γ,preferences(m).δ)
    @test shared(native)==(3.0,0.1)
    @test shared(SkibaModel(γ=3.0,δ=0.1))==(3.0,0.1)
    # A discount factor is not a rate, so neither domain reports the other's.
    @test !haskey(preferences(native),:ρ)
    @test !haskey(preferences(SkibaModel()),:discount)
end

@testset "Numerical bounds that exclude a steady state" begin
    uncovered(m,bounds)=solve(m,StateGrid(m;k=(n=100,bounds=bounds))).diagnostics.boundary.uncovered_steady_states
    skiba=DiscreteSkibaModel()
    cost=1/skiba.discount-1+skiba.depreciation
    low=(skiba.α*skiba.A_L/cost)^(1/(1-skiba.α))
    high=(skiba.α*skiba.A_H/cost)^(1/(1-skiba.α))+skiba.κ
    # kmax is below the escape threshold: no choice reaches it and the fit converges.
    truncated=StateGrid(skiba;k=(n=100,bounds=(0.01,3.0)))
    fit=solve(skiba,truncated)
    @test fit.converged
    @test !any(fit.diagnostics.boundary.upper_choice)
    @test collect(fit.diagnostics.boundary.uncovered_steady_states)≈[high]
    @test collect(uncovered(skiba,(1.5,8.0)))≈[low]
    @test isempty(uncovered(skiba,(0.01,8.0)))
    @test isempty(solve(skiba,StateGrid(skiba)).diagnostics.boundary.uncovered_steady_states)
    @test_throws ArgumentError (fit.diagnostics.boundary.uncovered_steady_states[1]=0.0)
    ramsey=DiscreteRamseyCassKoopmansModel()
    @test length(uncovered(ramsey,(0.01,2.0)))==1
    @test isempty(uncovered(ResourceModel(ramsey;transfer=0.1),(0.01,8.0)))
    @test length(uncovered(ResourceModel(ramsey;transfer=0.1),(0.01,2.0)))==1
    # A low productivity technology has no steady state on its own branch here.
    @test GrowthModels.capital_steady_states(DiscreteSkibaModel(κ=0.0),1.0)≈[(skiba.α*skiba.A_H/cost)^(1/(1-skiba.α))]
    # Every represented shock and type contributes, including the highest.
    types=DiscreteRamseyCassKoopmansModel(productivity=MarkovChain(values=[0.8,1.2],transition=[0.9 0.1;0.1 0.9]),
        population=Population([0.9,1.1],[0.5,0.5]))
    @test isempty(solve(types,StateGrid(types)).diagnostics.boundary.uncovered_steady_states)
    @test length(uncovered(types,(0.01,3.2)))==2
end

@testset "Interpolation can split one steady state into absorbing nodes" begin
    # The model has one steady state, but at this resolution two neighbouring nodes
    # are absorbing. This pins the behaviour and its explanation, not a desired result.
    m=DiscreteRamseyCassKoopmansModel()
    state=(m.α*m.A/(1/m.discount-1+m.depreciation))^(1/(1-m.α))
    g=StateGrid(m;k=(n=1000,bounds=(0.05*state,1.5*state)))
    fit=solve(m,g)
    @test fit.converged
    message=try stationary(fit); "" catch err; sprint(showerror,err) end
    @test occursin("Multiple stationary classes",message)
    @test occursin("Capital ranges",message)
    @test occursin("interpolation artefacts",message)
    k=collect(g.k)
    absorbing=[k[i] for i in eachindex(k) if fit.policy.next_capital[i]==k[i]]
    @test length(absorbing)==2
    @test absorbing[2]-absorbing[1]≈k[3]-k[2]
    @test all(x->abs(x-state)<k[3]-k[2],absorbing)
    # An initial distribution still selects the class weights.
    initial=distribution(g;masses=fill(1/length(g),size(g)))
    @test sum(mass(stationary(fit,initial)))≈1
    # The neighbouring-node resolution returns one class.
    single=solve(m,StateGrid(m;k=(n=500,bounds=(0.05*state,1.5*state))))
    @test sum(mass(stationary(single)))≈1
end
