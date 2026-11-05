# Run with: julia --project=. examples/discrete_time.jl
using GrowthModels

# Native per-period parameters; existing capital can be consumed.
model = DiscreteSkibaModel(discount=0.96, depreciation=0.08,
    risk_aversion=2.0, A_H=0.6, A_L=0.4, α=0.3, κ=2.0)
grid = StateGrid(model; k=(n=500, bounds=(0.001, 20.0)))
fit = solve(model, grid; history=true)
require_converged(fit)

# Re-optimize at each continuous capital state, including across policy jumps.
capital = [2.0]
consumed = Float64[]
for period in 1:100
    state = (k=last(capital),)
    push!(consumed, consumption(fit, state))
    push!(capital, next_capital(fit, state))
end
println((residual=fit.diagnostics.residual_norm,
    budget_error=fit.diagnostics.budget_error, final_capital=last(capital)))
# Optional: using Plots; plot_model(fit); plot_diagnostics(fit)
