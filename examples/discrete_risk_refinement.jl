# Run with: julia --project=. examples/discrete_risk_refinement.jl
# Finite-chain comparison: both matrix evolution and simulation use the same
# Rouwenhorst chain. Only the simulation keeps capital continuous (no grid lottery).
using GrowthModels, Random, SparseArrays

function risk_case(nk,nz;upper=20.0,periods=40,individuals_per_type=1000,
        persistence=0.9,innovation_std=0.1)
    types=Population([0.9,1.1],[0.5,0.5])
    m=DiscreteSkibaModel(productivity=LogAR1(;persistence,innovation_std),population=types)
    proposed=StateGrid(m;k=(n=nk,bounds=(0.001,upper)),z=(n=nz,))
    # Align initial capital and the target exactly across refinements.
    g=StateGrid(m;k=sort!(unique!(vcat(copy(proposed.k),[3.2,4.0]))),z=proposed.z)
    elapsed=@elapsed fit=solve(m,g)
    require_converged(fit)
    p=zeros(size(g)); p[findfirst(==(3.2),g.k),(nz+1)÷2,:].=types.shares
    initial=distribution(g;masses=p)
    final=advance(Evolution(fit),initial,periods)
    grid_mean=expectation(final,s->s.k)
    target=vec([GrowthModels.grid_state(g,i).k>=4.0 for i in CartesianIndices(size(g))])
    survivor=vec(copy(p))
    P=copy(fit.transition)
    grid_wait=0.0
    for _ in 1:periods
        grid_wait+=sum(survivor)
        survivor=P'*survivor
        survivor[target].=0
    end
    grid_hit=1-sum(survivor)
    means,variances,hits,hitvariances=Float64[],Float64[],Float64[],Float64[]
    waits,waitvariances=Float64[],Float64[]
    for (index,a) in enumerate(types.values)
        rng=MersenneTwister(132+index)
        capital=Float64[]; reached=Float64[]; waiting=Float64[]
        for _ in 1:individuals_per_type
            path=simulate(fit,(k=3.2,z=g.z[(nz+1)÷2],ability=a),periods;rng)
            push!(capital,last(path.states).k)
            firsthit=findfirst(s->s.k>=4.0,path.states)
            push!(reached,firsthit !== nothing)
            push!(waiting,firsthit === nothing ? periods : firsthit-1)
        end
        mean=sum(capital)/length(capital); hit=sum(reached)/length(reached)
        push!(means,mean); push!(variances,sum((capital .- mean).^2)/(length(capital)-1))
        push!(hits,hit); push!(hitvariances,sum((reached .- hit).^2)/(length(reached)-1))
        wait=sum(waiting)/length(waiting)
        push!(waits,wait); push!(waitvariances,sum((waiting .- wait).^2)/(length(waiting)-1))
    end
    mc_mean=sum(types.shares.*means)
    mc_hit=sum(types.shares.*hits)
    mean_se=sqrt(sum(types.shares.^2 .* variances)/individuals_per_type)
    hit_se=sqrt(sum(types.shares.^2 .* hitvariances)/individuals_per_type)
    classes,_=GrowthModels.closed_communicating_classes(sparse(P'))
    invariant=stationary(fit,initial)
    return (nk,nz,upper,grid_mean,mc_mean,mean_se,grid_hit,mc_hit,hit_se,
        length(classes),expectation(invariant,s->s.k),nnz(P),Base.summarysize(P),elapsed,
        fit.diagnostics.residual_norm,persistence,innovation_std,grid_wait,sum(types.shares.*waits),
        sqrt(sum(types.shares.^2 .* waitvariances)/individuals_per_type))
end

println("nk,nz,kmax,grid_mean,simulation_mean,mean_se,grid_hit,simulation_hit,hit_se,closed_classes,stationary_mean,transition_nnz,transition_bytes,solve_seconds,residual,persistence,innovation_std,grid_restricted_wait,simulation_restricted_wait,wait_se")
for (nk,nz,upper) in ((100,5,20.0),(200,5,20.0),(400,5,20.0),
        (200,3,20.0),(200,7,20.0),(300,5,30.0))
    println(join(risk_case(nk,nz;upper),","))
    flush(stdout)
end

# Hold stationary log variance fixed while changing persistence.
for persistence in (0.5,0.95)
    innovation_std=0.1*sqrt((1-persistence^2)/(1-0.9^2))
    println(join(risk_case(200,5;persistence,innovation_std),","))
    flush(stdout)
end
