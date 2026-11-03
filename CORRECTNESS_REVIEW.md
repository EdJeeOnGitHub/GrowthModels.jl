# Correctness follow-up to the external review

The external review was checked against `614172c`. This pass addresses
scientific correctness and misleading failure modes; the larger API redesign,
starting-guess experiments, threading, and further memory tuning are deferred.
The original review scripts are preserved in `test/review-reproductions/`.

## Changes that affect numerical results

| Finding | Resolution |
|---|---|
| False HJB convergence for γ<1 | Require a small value update **and** a finite absolute nonlinear residual `u(c) + A*v - ρ*v < crit` in maximum norm. Recompute the policy from the candidate value before this check. Policies using floored marginal values cannot certify convergence. Default pseudo-time step is 1 for γ<1, 1000 otherwise; explicit `Delta` is honored. |
| Truncated OU defaults | Use `exp(±4σ/sqrt(2θ))` by default, configurable with `z_width`. This matches the model's zero-mean OU process in log productivity. |
| Observation-dependent time accuracy | Real-time evolution defaults to a maximum step of 0.1. Explicit `dt=nothing` retains one step per observation interval. Substep counting accounts for floating-point errors in observation-time subtraction. |
| High-type capital bounds | Parametric stochastic grids use the highest productivity and ability to set the capital reference, followed by `kmax_f`. General nonparametric production still requires user assessment of capital bounds. |
| Missing transfer in interpolated capital drift | Store `flat_transfer` with solved policy variables and include it in deterministic, stochastic, and ability-model drift closures. Moving it into model types is deferred to the API redesign. |
| Incorrect stochastic HJB diagnostics | `V_err` now uses the entire generator, including productivity transitions. |
| Incorrect Skiba marginal product | Differentiate the active production branch, with the income floor respected in Poisson models. Scalar and array inputs are tested. |
| Misleading `k_star` interpretation | Add `technology_crossing` and document `k_star` as a compatibility alias for Skiba models. Plot labels identify the technology crossing. No optimal Skiba threshold is claimed. |
| Boundary checks | Validate every lower-capital boundary state, including intermediate shock/ability combinations. Use the floored zero-drift derivative consistently at boundaries. See the qualification below. |

Default-based results should be regenerated. Wider grids also reduce resolution
at fixed point counts; changes in bounds must be accompanied by refinement
checks. A small discrete residual establishes that the discrete equation was
solved, not that the discretization is accurate enough for every application.

A fixed maximum time step limits the influence of observation spacing. It does
not make arbitrary irregular observation schedules identical: observations that
fall between integration points require a different subdivision. Reduce `dt`
and compare the quantities of interest to establish time accuracy. Integer `T`
keeps its historical `0:T-1` observations and `implicit_steps` control.

## Smaller correctness fixes

- Poisson default grids derive their support and number of states from the
  process. A conflicting `Nz` gives an explicit error. Ability grids include
  their ability dimension.
- Additive Poisson income is no longer added to steady-state **capital** reference
  formulas. These references are not exact stochastic optimal steady states.
- SmoothSkiba's high steady-state approximation no longer adds κ. Its sigmoid
  derivative remains finite for steep transitions such as β=500.
- Deterministic `ydot` uses marginal production times capital drift.
- OU samplers allocate from the actual time-vector length, avoiding mismatches
  when `T/dt` is not integral. They retain the historical convention of returning
  points on `0:dt:T`; the last point can precede T. The SDE comment now uses σ dW.
- `solve_growth_model` without an explicit guess uses the solver's default guess.
- Automatic block solves inspect cross-block generator entries and fall back to
  a full sparse solve when a trailing state switches. Explicitly requested
  incompatible workspace blocks still fail rather than discard transitions.
- `normalize=false` evolution and its stationary result preserve the same total
  mass. The standalone matrix-taking stationary API still returns a probability
  density with unit mass.
- Failed HJB solves produce an explanatory `ArgumentError` when passed to
  `SolvedModel`, rather than a missing-method error.
- State construction promotes coordinate and production types together, enabling
  differentiation on fixed Float64 grids. State indexing uses the promoted type.
- Previously disabled differentiation assertions are restored. A policy-assembly
  test now inspects assembly directly instead of forcing convergence with an
  infinite tolerance.

## Qualifications and remaining limits

**Negative net output at high capital is not automatically a state-constraint
violation.** Capital can drift inward there while consumption remains positive.
Requiring `minimum(y - δk) > 0` everywhere would reject valid problems. The lower
boundary does need feasible nonnegative capital drift; it is now checked across
all other states. A regression covers a valid negative-upper-net-output case
with fractional γ, as well as an invalid intermediate lower-boundary state.
The upwind HJB/state-constraint framework is described in the
[Achdou et al. numerical appendix](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT_appendix.pdf).

**γ=1 is not newly supported.** It now raises an explicit error identifying the
unsupported log-utility case instead of failing later with a misleading initial
value message. Supporting a differentiable utility normalization through γ=1
requires a separate design decision.

**SmoothSkiba steady-state helpers are still approximations.** They assume the
sigmoid weight is near zero or one. The legacy SmoothSkiba `k_star` helper also
must not be interpreted as a solved optimal threshold.

**The two-asset model remains incomplete and unsupported.** This pass does not
invent missing model equations. Removing it from the exported surface, moving
Plots into an extension, replacing type piracy, and introducing new grid/result/
evolution types remain part of the deferred API work.

## Numerical checks

On the corrected defaults, the reproduced γ=0.5 case has maximum consumption
0.809569 and residual 9.85e-7 at the default `crit=1e-6`. The previous version
reported convergence with consumption around 2.03e31.

For the analytic RCK case γ=α=0.3, interior maximum consumption error falls from
6.51e-4 at 250 capital points to 1.62e-4 at 1000 points. Both nonlinear residuals
are below 1e-9. The expected analytic policy is
`c(k) = [(ρ + δ(1-α))/α] k`.

For the default OU parameters, the stationary targets are approximately
`E[z]=1.02401` and `sd(log z)=0.21784`. With the corrected four-standard-deviation
bounds:

| Productivity points | Mean z | SD log z |
|---:|---:|---:|
| 40 | 1.02159 | 0.23038 |
| 100 | 1.02305 | 0.22266 |
| 400 | 1.02375 | 0.21895 |

This both removes the previous severe truncation and makes the remaining
finite-grid error visible. The default 40-point grid is not a precision guarantee.

The maintained review regression suite currently has 95 assertions covering
these cases, mass consistency, observation schedules, derivatives, transfers,
state boundaries, and failure contracts. The complete suite passes **432,091 assertions**, including the 95 new review
checks and 31 restored derivative assertions. Validation command:
`GKSwstype=100 julia --project=. test/runtests.jl`.
