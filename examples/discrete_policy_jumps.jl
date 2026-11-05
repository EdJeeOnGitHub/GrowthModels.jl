# Run with: julia --project=. examples/discrete_policy_jumps.jl
# Lists the jumps in the default Skiba policy. Each jump's two choices straddle the
# previous jump point, and the first straddles the production crossing: optimal
# paths never land on a convex kink of the value function.
using GrowthModels

model=DiscreteSkibaModel()
println("crossing=",technology_crossing(model))
println("n,k,next_capital_below,next_capital_above,jump,reverses_direction")
for n in (1000,2000,4000,8000)
    fit=solve(model,StateGrid(model;k=(n=n,bounds=(0.001,8.0))))
    require_converged(fit)
    for j in policy_jumps(fit)
        println(join((n,j.k,j.next_capital_below,j.next_capital_above,j.jump,j.reverses_direction),","))
    end
    flush(stdout)
end

# With shocks the largest jumps persist under refinement, but their number grows.
shock=MarkovChain(values=[0.8,1.25],transition=[0.85 0.15;0.25 0.75])
risky=DiscreteSkibaModel(productivity=shock,population=Population([0.8,1.2],[0.5,0.5]))
println("n,z,ability,count_above_one_cell,count_all,largest_five_k_and_jump")
for n in (500,1000,2000,4000)
    fit=solve(risky,StateGrid(risky;k=(n=n,bounds=(0.01,12.0))))
    require_converged(fit)
    for (z,ability) in ((0.8,0.8),(1.25,1.2))
        kept=policy_jumps(fit;z,ability)
        everything=policy_jumps(fit;z,ability,minimum_jump=0)
        top=sort(everything;by=j->-j.jump)[1:min(5,length(everything))]
        println(join((n,z,ability,length(kept),length(everything),
            join((string(round(j.k,digits=3),":",round(j.jump,digits=3)) for j in top)," ")),","))
    end
    flush(stdout)
end
