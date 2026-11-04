# Run with: julia --project=. examples/discrete_policy_jumps.jl
# Lists every upward jump in the default Skiba policy on a fine query lattice. Each
# jump's two actions straddle the previous jump point (the first straddles the
# production crossing): optimal paths never land on a convex kink of the value.
using GrowthModels

model=DiscreteSkibaModel()
println("crossing=",technology_crossing(model))
println("n,k_at_jump,action_before,action_after,jump")
for n in (2000,4000,8000)
    grid=StateGrid(model;k=(n=n,bounds=(0.001,8.0)))
    fit=solve(model,grid)
    require_converged(fit)
    spacing=grid.k[3]-grid.k[2]
    queries=range(0.5,4.5;length=40001)
    actions=[next_capital(fit,(k=x,)) for x in queries]
    for i in 1:length(actions)-1
        jump=actions[i+1]-actions[i]
        jump>5spacing && println(join((n,queries[i],actions[i],actions[i+1],jump),","))
    end
    flush(stdout)
end
