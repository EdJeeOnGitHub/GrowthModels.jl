using Test, GrowthModels, SparseArrays, ForwardDiff

struct ReviewedProcess <: StochasticProcess
    rate::Float64
end
GrowthModels.generator(p::ReviewedProcess,z::AbstractVector) = sparse([-p.rate p.rate;2p.rate -2p.rate])
struct ReviewedModel{T,P} <: StochasticModel{T,P}
    economic_parameters::NamedTuple
    stochasticprocess::P
end
GrowthModels.preferences(m::ReviewedModel) = m.economic_parameters
GrowthModels.production(::ReviewedModel,s::NamedTuple) = .6*s.z*s.k^.3
GrowthModels.resources(m::ReviewedModel,s::NamedTuple) = production(m,s)+.1
GrowthModels.default_grid_spec(::ReviewedModel) = (k=(n=70,bounds=(.1,5.)),z=(n=2,bounds=(.8,1.2)))

@testset "Economic and process extension contracts" begin
    m=ReviewedModel{Float64,ReviewedProcess}((γ=2.,ρ=.05,δ=.05),ReviewedProcess(.1))
    g=StateGrid(m)
    fit=solve(m,g)
    @test fit.converged
    @test minimum(fit.policy.resources) > .1
    @test capital_drift(fit,(k=g.k[20],z=.8)) ≈ fit.policy.drift[20,1]
    state=(k=1.,z=.8)
    @test resources(ResourceModel(m),state) == resources(m,state)
    adjusted=ResourceModel(ResourceModel(m;transfer=.2,income_floor=.8);transfer=.1,income_floor=.5)
    @test resources(adjusted,state) == max(max(resources(m,state)+.2,.8)+.1,.5)
    @test preferences(ResourceModel(m)) == preferences(m)
    @test solve(ResourceModel(m),g).converged
    @test StateGrid(ResourceModel(m)).k == g.k
    initial=distribution(g;masses=fill(1/length(g),size(g)))
    @test sum(mass(stationary(fit,initial))) ≈ 1
    function moment(rho)
        model=ReviewedModel{typeof(rho),ReviewedProcess}((γ=2.,ρ=rho,δ=.05),ReviewedProcess(.1))
        r=solve(model,g;value_tolerance=1e-9,residual_tolerance=1e-9)
        expectation(stationary(r,initial),s->s.k)
    end
    h=1e-5
    @test ForwardDiff.derivative(moment,.05) ≈ (moment(.05+h)-moment(.05-h))/(2h) rtol=1e-4
    @test_throws DimensionMismatch solve(m,StateGrid(m;z=[.8,1.,1.2]))
    for process in (OrnsteinUhlenbeckProcess(θ=.1,σ=.1),PoissonProcess(z=[.8,1.2],Q=[-.1 .1;.2 -.2]))
        base=StochasticRamseyCassKoopmansModel(process)
        grid=StateGrid(base;k=(n=5,),z=process isa PoissonProcess ? [.8,1.2] : collect(range(.8,1.2;length=5)))
        st,hp,Q=GrowthModels.numerical_state(base,grid)
        @test Q ≈ GrowthModels.construct_diffusion_matrix(process,st,hp)
    end
end

@testset "Keyword constructors promote calibration parameters" begin
    process=OrnsteinUhlenbeckProcess(θ=.1,σ=.1)
    constructors=(x->RamseyCassKoopmansModel(ρ=x),x->SkibaModel(ρ=x),
        x->SmoothSkibaModel(ρ=x),x->StochasticRamseyCassKoopmansModel(process;ρ=x),
        x->StochasticSkibaModel(ρ=x),x->StochasticSkibaAbilityModel(ρ=x),
        x->StochasticNPAbilityModel(ρ=x))
    for constructor in constructors
        @test ForwardDiff.derivative(x->preferences(constructor(x)).ρ,.05)==1
    end
end
