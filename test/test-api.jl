using Test, GrowthModels, LinearAlgebra, SparseArrays, ForwardDiff, Serialization
const GM = GrowthModels

@testset "Economic and grid contracts" begin
    m = ResourceModel(StochasticSkibaAbilityModel(); transfer=.2, income_floor=.5,
        population=Population([.8,1.2],[.3,.7]))
    g = StateGrid(m;k=(n=40,bounds=(.1,8.0)),z=(n=8,))
    reordered = StateGrid(m;z=(n=8,),k=(n=40,bounds=(.1,8.0)))
    @test size(g) == (40,8,2)
    @test GM.compatible_grid(g,reordered)
    @test resources(m,(k=.1,z=.5,ability=.8)) == .5
    @test resources(m,(k=4.,z=1.,ability=1.2)) ≈ max(production(m,(k=4.,z=1.,ability=1.2))+.2,.5)
    @test_throws ArgumentError StateGrid(m; ability=[.7,1.])
    @test_throws ArgumentError Population([1.,2.],[.3,.3])
    original = g.k
    @test_throws ArgumentError g.k[1] = 9
    @test g.k == original
    @test size(cell_weights(g)) == size(g)
    @test sum(cell_weights(g)[:,1,1]) ≈ last(g.k)-first(g.k)
    @test coordinates(g).ability[1,1,:] == [.8,1.2]
    @test_throws MethodError g[:k]
end

@testset "Solutions, failure and reproducibility" begin
    m = RamseyCassKoopmansModel()
    g = StateGrid(m;k=(n=100,))
    fit = solve(m,g;history=true)
    @test fit.converged && fit.diagnostics.admissible
    @test fit.diagnostics.residual_norm < fit.settings.residual_tolerance
    @test length(fit.diagnostics.history) == fit.diagnostics.iterations
    @test size(fit.initial_value) == size(g)
    replay = solve(m,g;initial=fit.initial_value,Delta=fit.settings.Delta)
    @test replay.value == fit.value
    warm = solve(m,g;initial=fit)
    @test warm.diagnostics.iterations == 1
    @test warm.initial_value == fit.value
    @test warm.initial_value !== fit.value
    failed = solve(m,g;maxiters=0)
    @test !failed.converged && failed.diagnostics.stop_reason == :iteration_limit
    @test size(failed.policy.consumption) == size(g)
    @test all(isfinite,failed.policy.consumption)
    @test_throws ArgumentError require_converged(failed)
    @test_throws ArgumentError consumption(failed,(k=1.,))
    @test_throws ArgumentError Evolution(failed;dt=.1)
    @test_throws ArgumentError stationary(failed)
    @test_throws ArgumentError solve(m,g;initial=fill(NaN,size(g)))
    @test_throws DimensionMismatch solve(m,g;initial=zeros(3))
    @test_throws ArgumentError solve(m,g;Delta=0)
    r = ResourceModel(m; transfer=.2)
    f = solve(r,g)
    for i in (2,40,90)
        state = (k=g.k[i],)
        @test capital_drift(f,state) ≈ f.policy.drift[i]
        @test consumption(f,state) ≈ f.policy.consumption[i]
    end
    tighter = StateGrid(m;k=range(g.k[2],g.k[end-1];length=50))
    rg = regrid(fit,tighter)
    @test solve(m,tighter;initial=rg).converged
    @test_throws ArgumentError solve(m,tighter;initial=fit)
    bigger = StateGrid(m;k=range(g.k[1],g.k[end]*1.1;length=50))
    @test_throws ArgumentError regrid(fit,bigger)
    @test size(regrid(fit,bigger;extrapolate=true).value) == size(bigger)
    io = IOBuffer(); serialize(io,fit); seekstart(io)
    restored = deserialize(io)
    @test restored.initial_value == fit.initial_value
    @test restored.settings == fit.settings
end

@testset "Probability and observation contracts" begin
    m = StochasticRamseyCassKoopmansModel()
    g = StateGrid(m;k=(n=60,),z=(n=10,))
    fit = solve(m,g)
    p = fill(1/length(g),size(g))
    d = distribution(g;masses=p)
    @test mass(StateDistribution(g, view(p, ntuple(_ -> Colon(), ndims(p))...))) == p
    @test_throws ArgumentError distribution(g;masses=2p)
    @test_throws ArgumentError distribution(g;masses=zeros(size(g)))
    @test_throws ArgumentError distribution(g;masses=-p)
    @test_throws ArgumentError distribution(g;masses=p,density=p)
    @test_throws DimensionMismatch distribution(g;masses=vec(p))
    @test mass(distribution(g;density=density(d))) ≈ p
    mass(d)[1] = 0
    @test mass(d) == p
    @test expectation(d,s->1.) ≈ 1
    @test sum(values(boundary_mass(d))) ≈ 2/size(g)[1]
    e = Evolution(fit;dt=.1)
    @test_throws UndefKeywordError Evolution(fit)
    @test_throws ArgumentError Evolution(fit;dt=0)
    times = [0.,.13,.7,1.03,2.]
    path = trajectory(e,d,times)
    dense = trajectory(e,d,sort(unique(vcat(times,collect(0.:.07:2.)))))
    for t in times
        @test mass(path,t) == mass(dense,t)
        @test sum(mass(path,t)) ≈ 1 atol=1e-12
        @test minimum(mass(path,t)) >= 0
    end
    @test mass(d) == p
    @test_throws ArgumentError mass(path,.14)
    @test mass(advance(e,d,2.)) == mass(final(path))
    moment = observe(e,d,times,(k=s->s.k,k2=s->s.k^2))
    @test moment.values.k ≈ expectation(path,s->s.k)
    @test moment.values.k2 ≈ expectation(path,s->s.k^2)
    shifted = trajectory(e,d,times .+ 5;t0=5.)
    @test mass(final(shifted)) ≈ mass(final(path))
    delayed = trajectory(e,d,[1.03,2.])
    @test mass(final(delayed)) == mass(final(path))
    @test_throws ArgumentError trajectory(e,d,[0.,0.])
    @test_throws ArgumentError trajectory(e,d,[-1.,1.])
    sd = stationary(fit,d)
    @test sum(mass(sd)) ≈ 1
    @test maximum(abs.(transpose(fit.generator)*vec(mass(sd)))) < 1e-9
    @test mass(stationary(fit)) ≈ mass(sd)
    # Linear interpolation of actual lattice states, not a shortened step.
    lattice = trajectory(e,d,[.1,.2])
    @test mass(path,.13) ≈ .7mass(lattice,.1)+.3mass(lattice,.2)
end

@testset "Ability classes and functional resource rules" begin
    pop = Population([.8,1.2],[.3,.7])
    base = FunctionalModel((k,z,a)->.6*k^.3*z*a,OrnsteinUhlenbeckProcess(θ=.1,σ=.1),pop)
    m = ResourceModel(base;transfer=.1,income_floor=.3)
    g = StateGrid(m;k=(n=50,bounds=(.01,8.)),z=(n=10,))
    fit = solve(m,g)
    @test fit.converged
    p = zeros(size(g)); p[:,:,1] .= .3/(50*10); p[:,:,2] .= .7/(50*10)
    d = distribution(g;masses=p)
    @test vec(sum(mass(stationary(fit));dims=(1,2))) ≈ [.3,.7]
    for result in (stationary(fit,d),advance(Evolution(fit;dt=.1),d,3.))
        @test vec(sum(mass(result);dims=(1,2))) ≈ [.3,.7] atol=1e-11
    end
    @test fit.policy.resources == [resources(m,GM.grid_state(g,i)) for i in CartesianIndices(size(g))]
end

@testset "Scientific regressions" begin
    for gamma in (.3,.5,.9,1.5,2.)
        m = RamseyCassKoopmansModel(γ=gamma)
        fit = solve(m,StateGrid(m;k=(n=100,)))
        @test fit.converged
        @test maximum(fit.policy.consumption) < 10
        @test fit.diagnostics.residual_norm < 1e-6
    end
    @test_throws ArgumentError solve(RamseyCassKoopmansModel(γ=1.),StateGrid(RamseyCassKoopmansModel();k=(n=20,)))
    errors = Float64[]
    for n in (250,1000)
        m=RamseyCassKoopmansModel(γ=.3,α=.3)
        kss=GM.k_steady_state(m)
        f=solve(m,StateGrid(m;k=(n=n,bounds=(.3kss,1.5kss)));value_tolerance=1e-9,residual_tolerance=1e-9,maxiters=2000)
        @test f.converged
        phi=(m.ρ+m.δ*(1-m.α))/m.α
        push!(errors,maximum(abs.(f.policy.consumption[4:end-3]-phi*f.grid.k[4:end-3])))
    end
    @test errors[2] < errors[1] < 1e-3
    m=StochasticRamseyCassKoopmansModel(); g=StateGrid(m;k=(n=2,),z=(n=400,))
    D=GM.exogenous_generator(m,g); Q=D[1:2:end,1:2:end]
    p=GM.stationary_mass(sparse(Q'))
    sigma=m.stochasticprocess.σ/sqrt(2m.stochasticprocess.θ)
    @test dot(g.z,p) ≈ exp(sigma^2/2) atol=.002
    @test sqrt(dot(log.(g.z).^2,p)-dot(log.(g.z),p)^2) ≈ sigma atol=.005
    for model in (SkibaModel(),SmoothSkibaModel(β=500.))
        for k in (.01,2.2,3.)
            @test isfinite(marginal_product(model,(k=k,)))
            @test marginal_product(model,(k=k,)) ≈ ForwardDiff.derivative(x->production(model,(k=x,)),k)
        end
    end
end
