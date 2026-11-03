using Test, GrowthModels, SparseArrays, Serialization, ForwardDiff

function scalar_inspection(fit)
    s=0.0
    for i in 1:1000
        s+=fit.value[mod1(i,length(fit.value))]
    end
    s
end
@testset "Read-only snapshots" begin
    p=PoissonProcess(z=[.8,1.2],Q=[-.1 .1;.2 -.2])
    m=StochasticRamseyCassKoopmansModel(p)
    g=StateGrid(m;k=(n=60,bounds=(.1,5.)))
    f=solve(m,g;history=true)
    require_converged(f)
    state=(k=g.k[20],z=.8)
    c=consumption(f,state); drift=capital_drift(f,state)
    for a in (f.value,f.initial_value,f.policy.consumption,f.policy.drift,f.generator,
              f.diagnostics.history,g.k,g.z)
        @test_throws ArgumentError setindex!(a,first(a),firstindex(a))
        @test_throws ArgumentError fill!(a,first(a))
        @test_throws ArgumentError view(a,:)[1] = first(a)
        editable=copy(a)
        editable[firstindex(editable)]=first(a)
        @test editable == a
    end
    @test_throws ArgumentError f.value .= 0
    @test_throws ArgumentError parent(f.value)[1] = 0
    @test_throws ArgumentError f.value.storage
    @test sparse(f.generator) * vec(f.value) ≈ f.generator * vec(f.value)
    @test sum(transpose(f.generator)*ones(length(g))) ≈ 0 atol=1e-9
    retrieved=f.model
    retrieved.stochasticprocess.z[1]=.9
    p.z[1]=.7
    @test consumption(f,state) == c
    @test capital_drift(f,state) == drift
    @test f.model.stochasticprocess.z[1] == .8
    d=distribution(g;masses=fill(1/length(g),size(g)))
    @test_throws ArgumentError d.probabilities[1] = 2
    mass(d)[1]=2
    @test expectation(d,s->1.) ≈ 1
    e=Evolution(f;dt=.1)
    @test_throws ArgumentError e.solver
    @test_throws ArgumentError e.generator
    path=trajectory(e,d,[0.,.1,.25])
    @test_throws ArgumentError path.times[1] = 1
    @test_throws ArgumentError path.probabilities[1] = 2
    obs=observe(e,d,[0.,.1],(k=s->s.k,))
    @test_throws ArgumentError obs.values.k[1] = 1
    pop=Population([.8,1.2],[.3,.7])
    @test_throws ArgumentError pop.shares[1] = .5
    scalar_inspection(f)
    @test @allocated(scalar_inspection(f)) < 4096
    io=IOBuffer();serialize(io,f);seekstart(io)
    restored=deserialize(io)
    @test restored.value == f.value
    @test_throws ArgumentError restored.value[1]=0
    @test solve(f.model,g;initial=f).converged # m's process was mutated above: restore first
end
