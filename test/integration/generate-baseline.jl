# Run with the environment of commit f711991 (before the API redesign).
# This intentionally uses its retired API to regenerate the comparison fixture.
using GrowthModels, SparseArrays
const G=GrowthModels
p=PoissonProcess(z=[0.0,.1],Q=[-.2 .2;.3 -.3])
models=(rck=RamseyCassKoopmansModel(),skiba=SkibaModel(),smooth=SmoothSkibaModel(),
    ou=StochasticRamseyCassKoopmansModel(),poisson=StochasticSkibaModel(p),ability=StochasticSkibaAbilityModel())
open(joinpath(@__DIR__, "..", "test-data", "api-baseline.csv"),"w") do io
    println(io,"case,index,value,consumption,final_mass")
    for (name,m) in pairs(models)
        k=HyperParams(N=60,xmin=.01,xmax=5.)
        hp=if m isa G.DeterministicModel
            StateSpaceHyperParams((k=k,))
        elseif name==:poisson
            StateSpaceHyperParams((k=k,z=HyperParams(N=2,xmin=0.,xmax=.1)))
        elseif name==:ability
            StateSpaceHyperParams((k=k,z=HyperParams(N=6,xmin=.8,xmax=1.2),η=HyperParams(N=2,xmin=.8,xmax=1.2)))
        else
            StateSpaceHyperParams((k=k,z=HyperParams(N=6,xmin=.8,xmax=1.2)))
        end
        result=solve_HJB(m,hp;verbose=false)
        @assert result.value.convergence_status
        dims=size(result.value.v); weights=create_dx_stacked(vec(result.variables.k[:,1,1]),prod(dims)÷dims[1])
        probability=fill(1/prod(dims),prod(dims))
        evo=StateEvolution(probability ./ weights,sparse(result.value.A'),[0.,1.],dims,weights;dt=.25,normalize=false)
        for i in eachindex(probability)
            println(io,name,",",i,",",result.value.v[i],",",result.variables.c[i],",",vec(evo.cell_masses)[length(probability)+i])
        end
    end
end
