using Test, GrowthModels
@testset "Observation interpolation refinement" begin
    for model in (RamseyCassKoopmansModel(),StochasticRamseyCassKoopmansModel())
        grid=model isa DeterministicModel ? StateGrid(model;k=(n=120,)) : StateGrid(model;k=(n=60,),z=(n=12,))
        fit=solve(model,grid)
        for concentrated in (false,true)
            p=zeros(size(grid))
            if concentrated
                p[div(length(p),3)]=1
            else
                p .= 1/length(p)
            end
            initial=distribution(grid;masses=p)
            times=[.135,.575,1.035]
            obs=(k=s->s.k,k2=s->s.k^2)
            ref=observe(Evolution(fit;dt=.001),initial,times,obs)
            coarse=observe(Evolution(fit;dt=.1),initial,times,obs)
            fine=observe(Evolution(fit;dt=.025),initial,times,obs)
            for key in keys(obs)
                @test maximum(abs.(fine.values[key]-ref.values[key])) < maximum(abs.(coarse.values[key]-ref.values[key]))
            end
        end
    end
end
