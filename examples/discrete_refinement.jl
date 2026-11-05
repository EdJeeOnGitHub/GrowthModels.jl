# Run with: julia --project=. examples/discrete_refinement.jl
# This experiment follows the jump where capital reverses direction, which this
# calibration has exactly once.
using GrowthModels

function endpoint(fit,k;periods=300)
    for _ in 1:periods
        k=next_capital(fit,(k=k,))
    end
    return k
end

model=DiscreteSkibaModel()
println("n_requested,n_actual,kmax,crossing_inserted,threshold,left_action,right_action,low_path_end,high_path_end,residual,boundary_choices")
for (n,upper,insert_crossing) in ((250,20.0,true),(500,20.0,true),
        (1000,20.0,true),(2000,20.0,true),(600,12.0,true),
        (1500,30.0,true),(1000,20.0,false))
    grid=insert_crossing ? StateGrid(model;k=(n=n,bounds=(0.001,upper))) :
        StateGrid(model;k=collect(range(0.001,upper;length=n)))
    fit=solve(model,grid)
    require_converged(fit)
    escape=only(filter(j->j.reverses_direction,policy_jumps(fit)))
    threshold,left,right=escape.k,escape.next_capital_below,escape.next_capital_above
    low=endpoint(fit,threshold-0.05)
    high=endpoint(fit,threshold+0.05)
    boundary=count(fit.diagnostics.boundary.lower_choice)+count(fit.diagnostics.boundary.upper_choice)
    println(join((n,length(grid),upper,insert_crossing,threshold,left,right,
        low,high,fit.diagnostics.residual_norm,boundary),","))
end
