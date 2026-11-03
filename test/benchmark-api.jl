# Run under either the pre-redesign checkout or this checkout's environment:
# julia --project=/path/to/checkout /path/to/this/test/benchmark-api.jl
using GrowthModels, Statistics, Printf
const GM = GrowthModels
const modern = isdefined(GM, :GrowthSolution)
function measure(f)
    f() # compile
    samples = [@timed(f()) for _ in 1:3]
    return median(s.time for s in samples), minimum(s.bytes for s in samples)
end
function setup(name)
    m = name == :skiba ? SkibaModel() : StochasticSkibaModel()
    n = name == :skiba ? 1000 : 200
    if modern
        grid = name == :skiba ? StateGrid(m;k=(n=n,bounds=(.01,8.))) :
            StateGrid(m;k=(n=n,bounds=(.01,8.)),z=(n=20,bounds=(.5,1.5)))
        solver = () -> GM.solve(m,grid)
        fit = solver(); require_converged(fit)
        d = distribution(grid;masses=fill(1/length(grid),size(grid)))
        e = Evolution(fit;dt=.1)
        evolve = () -> trajectory(e,d,0.:1.:20.)
        return solver,evolve,fit.diagnostics.iterations
    else
        k = GM.HyperParams(N=n,xmin=.01,xmax=8.)
        hp = name == :skiba ? GM.StateSpaceHyperParams((k=k,)) :
            GM.StateSpaceHyperParams((k=k,z=GM.HyperParams(N=20,xmin=.5,xmax=1.5)))
        solver = () -> GM.solve_HJB(m,hp;verbose=false)
        fit = solver()
        sm = GM.SolvedModel(m,fit)
        weights=GM.create_dx_stacked(vec(sm.variables.k[:,1]),length(fit.value.v)÷n)
        initial=fill(1/length(fit.value.v),length(fit.value.v))./weights
        evolve = () -> GM.StateEvolution(initial,sm,0.:1.:20.;dt=.1)
        return solver,evolve,fit.iter
    end
end
println("API=",modern ? "new" : "baseline"," source=",pathof(GM))
for name in (:skiba,:ou)
    solver,evolve,iterations=setup(name)
    for (operation,f) in ((:solve,solver),(:trajectory,evolve))
        seconds,bytes=measure(f)
        @printf("%s %s median_seconds=%.6f allocated_bytes=%d iterations=%d\n",name,operation,seconds,bytes,iterations)
    end
end
if modern
    model=StochasticSkibaModel()
    grid=StateGrid(model;k=(n=200,bounds=(.01,8.)),z=(n=20,bounds=(.5,1.5)))
    fit=GM.solve(model,grid)
    d=distribution(grid;masses=fill(1/length(grid),size(grid)))
    e=Evolution(fit;dt=.1)
    operations=(warm_solve=()->GM.solve(model,grid;initial=fit),
        moments=()->observe(e,d,0.:1.:20.,(k=s->s.k,)),
        final_only=()->advance(e,d,20.))
    for (name,f) in pairs(operations)
        seconds,bytes=measure(f)
        @printf("ou %s median_seconds=%.6f allocated_bytes=%d\n",name,seconds,bytes)
    end
end
