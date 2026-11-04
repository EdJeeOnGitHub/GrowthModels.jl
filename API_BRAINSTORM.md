# API brainstorm

Status: accepted design decisions. Implementation and acceptance evidence are
tracked in [API_IMPLEMENTATION.md](API_IMPLEMENTATION.md); current runnable
interfaces are documented in [README.md](README.md). The discussion below
retains the design rationale and the proposals that led to those decisions.

## Agreed direction

- **Breaking changes are allowed.** No compatibility wrappers or deprecation
  period are required. Update examples and downstream callers directly.
- **Return sampled observations.** Accept arbitrary observation times upfront;
  hide integration and interpolation. Results contain requested observations,
  not a continuously queryable solution.
- **Return inspectable non-convergence.** Failed solves return `converged=false`.
  Policy evaluation, simulation, and stationary calculations reject them. Raw
  arrays remain inspectable; `require_converged(fit)` provides an immediate error.
- **Require unit-mass probability input.** Reject nonunit input outside a small,
  documented floating-point tolerance. There is no constructor normalization
  switch and no silent renormalization during evolution.
- **Model exploration now, efficient structural calibration later.** Preserve
  warm starts, reusable evolution, moment calculations, and differentiation.
  Do not build an estimation framework as part of this redesign.

### Accepted decisions from the design review

1. Migrate GrowthCounterfactuals as part of implementing the new API, including
   its income-floor rule and callable production objects.
2. Use `solve(model, grid)`; omit a public `GrowthProblem` wrapper.
3. Use `resources(model, state)` as the shared economic extension point.
4. Keep explicitly supplied grids fixed. Rebuilding from a changed model is
   explicit; transferring a previous value guess to changed coordinates uses
   `regrid`.
5. Reject regridding extrapolation by default; allow an explicit opt-in.
6. Require unit-mass distributions and fail on nonunit input. Any normalization
   must be performed explicitly before construction using the grid measure.
7. Require `dt` in `Evolution(fit; dt=...)`.
8. Use separate `trajectory`, `advance`, and `observe` operations.
9. Reject unconverged fits in downstream calculations while preserving raw
   values, policies, and diagnostics for inspection.
10. Retain a copy of the actual starting value array and resolved solver settings
    in the result for reproducibility.

The public API should describe economics, numerical accuracy, and desired
outputs. Sparse layouts, factorization caches, and conversion buffers belong
inside the implementation.

## Review findings incorporated

Source inspection confirmed these downstream extension points in
`/home/ed/projects/aggregate-transfers/packages/GrowthCounterfactuals/src/`:

- `ModelSolver.jl` extends `GrowthModels.update_v` to compute
  `max(production + flat_transfer, min_y)`.
- It also extends the three-positional-argument `solve_HJB` for `StochasticModel`.
  This is more specific than this repo's `Model` method. If loaded against this
  version, it takes precedence and runs its own iteration loop. Its resolved
  runtime dependency/version and actual dispatch have not been verified here.
- `ProdFunctionInterp.jl` defines callable `InterpProdFunction(x, z, η)` backed
  by an interpolant. The new extension contract must support this use case.

Retiring solver overrides is acceptable only if economic customization has a
proper replacement. An additive transfer alone does not express an income floor.

The review's proposals for nonunit measures and public lattice-constrained
`advance!` are not adopted: they conflict with the probability-only scope and
our decision to defer incremental stepping. Its observation-error measurements
are useful external evidence, not independently reproduced API acceptance tests.

## A proposed everyday workflow

```julia
model = StochasticRamseyCassKoopmansModel(...)
grid = StateGrid(model; k=(n=1000,), z=(n=100, width=4))
fit = solve(model, grid)

fit.converged
fit.diagnostics.residual_norm
fit.settings
consumption(fit, (k=2.0, z=1.0))
capital_drift(fit, (k=2.0, z=1.0))

initial = distribution(grid; masses=initial_cell_probabilities)
evolution = Evolution(fit; dt=0.1)
path = trajectory(evolution, initial, 0.0:1.0:20.0; t0=0.0)
mean_k = expectation(path, state -> state.k)
long_run = stationary(fit, initial)
```

Use named state arguments consistently for policies and observables. A scalar
`k` shorthand for deterministic models can wait. Do not make a density function
of capital silently imply a joint distribution over productivity and ability.

## 1. Model owns economics; grid owns discretization

The **model** owns production, preferences, resource rules, processes, and the
population specification. Ability support and population shares belong here:
changing them changes the economic population, not merely numerical resolution.
An explicit initial distribution describes the joint population state; the
population specification provides defaults, not a reason to overwrite supplied
probabilities silently.

The **grid** owns realized coordinates, axis kinds, cell weights, and canonical
layout. Construct it from the model and resolution settings:

```julia
grid = StateGrid(model;
    k=(n=1000, bounds=(0.01, 12.0), spacing=:stretched),
    z=(n=100, width=4),
)
grid.k
grid.z
```

Proposed contracts:

- Named axes use properties such as `grid.k`. Keyword insertion order must not
  determine layout: `StateGrid(model; z=..., k=...)` means the same grid.
- The model's state schema defines canonical order, initially `(k, z, ability)`.
  Stacking is private. Validate supported schemas rather than guessing them.
- Axis metadata distinguishes capital quadrature from Markov-state probability.
  Initially density means density per unit capital and mass over Markov states,
  including a discretized OU process. Do not imply a joint continuous density.
- Grid objects retain both construction specifications and realized coordinates.
- Independent generator blocks are determined or validated from transitions,
  not inferred solely from the number or order of axes.
- Expose boundary mass and drift diagnostics. These help assess truncation but
  cannot certify that every model's grid is large enough.

Drop `GrowthProblem` from the initial proposal. A public wrapper containing only
model and grid does not yet hide enough complexity to justify another concept.
Use `solve(model, grid; ...)`; introduce a problem type later only if it acquires
substantial responsibilities.

### Parameter changes and warm starts

```julia
next_grid = StateGrid(next_model; k=capital_spec, z=productivity_spec)
guess = regrid(fit, next_grid)
next_fit = solve(next_model, next_grid; initial=guess)
```

Changing OU volatility changes coordinates derived from stationary width. Do not
reuse a grid derived from the old model without an explicit fixed-grid choice.

- For economic changes with unchanged geometry, `initial=fit` copies its value
  guess. It never reuses the old policy or generator.
- Compatibility checks include axis meaning and coordinates, not just shape.
- For changed coordinates, require explicit `regrid` with a documented
  interpolation/extrapolation policy. Default to rejecting extrapolation.
- Intentionally fixed coordinates are supported and recorded; the caller then
  owns their adequacy across parameter changes. No silent grid adaptation.

## 2. One solve result, including actual settings

Use a parametrically typed `GrowthSolution` with a stable field structure.

| Field | Meaning |
|---|---|
| `model` | Economic specification used |
| `grid` | Actual realized grid and construction specification |
| `value` | Returned finite value iterate |
| `policy` | Finite policy computed from that same iterate |
| `converged` | Whether all convergence requirements passed |
| `diagnostics` | Stop reason, iterations, update/residual norms, admissibility, optional history |
| `settings` | Resolved tolerances, max iterations, pseudo-time step, algorithm, and initialization strategy |
| `initial_value` | Owned copy of the actual starting value array, after any promotion or regridding |

Defaults must be resolved and recorded, including `Delta` and the initial-guess
strategy. Recording only keywords passed by the user is insufficient. For a
default, warm, or custom start, retain an owned copy of the actual starting value
array in `initial_value`; a label or hash alone is insufficient. Accept the cost
of one additional grid-sized array. Subsequent iteration or caller mutation must
not overwrite this snapshot. Adaptive methods must also record their relevant
settings and stop decisions.

Failure contract:

- Invalid input throws before iteration.
- Iteration-limit failure returns the last finite iterate and its corresponding
  policy, with `converged=false`.
- A nonfinite update returns the last finite value/policy pair with
  `stop_reason=:nonfinite`. If no such pair exists, throw an explanatory error.
- Finite does not necessarily mean admissible; expose that distinction in
  diagnostics and never mark an inadmissible solution as converged.
- Evolution and policy-evaluation helpers reject unconverged solutions. Raw
  result arrays remain inspectable without inventing fallback values.

```julia
fit = solve(model, grid; maxiters=20)
fit.converged || @show fit.diagnostics.stop_reason
require_converged(fit)  # explicit convenience for batch callers
```

Prefer this helper to a separate `on_failure` solve mode. A stable result
contract does not require hiding unrecoverable failures or fabricating policies.

## 3. StateDistribution: probability semantics and shapes

Use `StateDistribution` rather than `Distribution`, avoiding a name collision
with Distributions.jl. Attach the grid and store **cell probabilities** once.

```julia
initial = distribution(grid; masses=cell_probabilities)
initial = distribution(grid; density=unit_mass_density_values)

mass(initial)
density(initial)
expectation(initial, state -> state.k)
```

Proposed contracts:

- Exactly one of `masses` or `density` is supplied, as a shaped array matching
  the grid. Function-based constructors can be added later with a complete named
  state argument, never an implicit assumption about omitted axes.
- Require unit mass within a small, documented numerical tolerance. For density
  input, validate the weighted integral, not the unweighted sum of entries.
- Reject negative, nonfinite, zero-total-mass, and nonunit input. Report the
  measured total mass and tolerance in the error.
- Do not offer a `normalize` constructor keyword. Callers who want normalization
  perform it explicitly beforehand, using the grid measure. Accepted roundoff
  deviations are not silently rescaled to exactly one.
- `mass` and `density` return shaped copies in canonical axis order, e.g.
  `Nk × Nz × Nability`, never private stacked vectors or writable internal views.
- Grid axis properties return coordinate vectors; a separately specified helper
  can provide shaped coordinate arrays for downstream broadcasting.
- Distributions carry no time. Paths own observation times; calls specify their
  initial time separately.
- `expectation(d, f)` evaluates `f` on named states and computes its probability
  weighted sum. `expectation(path, f)` returns values at the saved times.
- No public `integral` or general nonunit measure type in the first release.

Choose numerical validation tolerances explicitly. A conservation failure during
stepping must not be concealed by automatic normalization.

## 4. Private evolution caches; separate output operations

```julia
evolution = Evolution(fit; dt=0.1)
path = trajectory(evolution, initial, times; t0=0.0)
last = advance(evolution, initial, 20.0; t0=0.0)
moments = observe(evolution, initial, times,
    (mean_k = state -> state.k, second_k = state -> state.k^2); t0=0.0)
long_run = stationary(fit, initial)
```

`Evolution` owns generator/grid snapshots, integration settings, factorization
caches, and buffers. Require `dt` explicitly in the initial API so the numerical
accuracy choice is visible. There is no public workspace or blocksize argument.
Reuse an evolution object across initial distributions; use a separate mutable
object for concurrent calls. Prefer constructing a new one for a new fit until
profiling demonstrates a need for validated reinitialization.

Use separate functions with stable output meanings instead of `save` modes:

| Operation | Output and storage |
|---|---|
| `trajectory` | All requested distributions and times, plus settings/diagnostics |
| `advance` | One final `StateDistribution`, no trajectory history |
| `observe` | Named expectations and times with settings/diagnostics, no distribution history |
| `stationary` | A limiting `StateDistribution`, independent of integration timestep |

All are nonmutating with respect to the initial distribution. `advance` is a
one-shot final-output request: its duration is relative to `t0`. It does not
promise compositional numerical continuation from interpolated endpoints.
A mutable, incremental integrator is a separate deferred feature.

`observe` initially computes named expectations of state functions. General
nonlinear functionals of a whole distribution would need a separately specified
interface; do not overload the same callback with two argument meanings.

`stationary(fit, initial)` uses the generator, not `Evolution` or `dt`. It
preserves the initial shares of closed classes (including transient absorption)
and returns unit mass. Omit `initial` only when the limiting probability
vector is uniquely determined. Reject unconverged fits here too.

## 5. Observation contract and APOSD rationale

Agreed: accept arbitrary observation times upfront, and return sampled outputs.

- Output times need not align with integration steps.
- Results expose exactly the requested observations. Lookup at an unrequested
  time raises “time was not requested”; it never silently interpolates saved
  snapshots or reruns the solver.
- Additional times require a new request, which may reuse private factors.
- With the same initial state, initial time, grid, and integration settings,
  changing output times must not alter results at shared times.
- Observation times are absolute and must be at or after explicit `t0` (default
  zero). The first requested output does not redefine the initial condition time.

This is a meaningful output abstraction, not a restriction based on solver
mechanics. Rejecting times because they disrupt internal factorizations would
leak implementation decisions. Requiring times because the product is a finite
collection of observations does not promise continuous evaluation. This follows
APOSD's information-hiding and deep-module principles; see
[Ousterhout's abstraction notes](https://web.stanford.edu/~ouster/CS349W/lectures/abstraction.html).

Proposed implementation:

1. Anchor a fixed internal integration lattice at `t0` with step `dt`.
2. Compute the bracketing lattice states for each observation, including the
   state after an off-lattice final observation. Do not shorten steps for output.
3. Interpolate cell masses linearly. Convex combinations preserve positivity and
   unit mass up to numerical error. Test accuracy with timestep refinement.
4. Keep only bracketing states and the operation's requested outputs.
5. Never restart the integrator from an interpolated observation internally.

The fixed lattice is chiefly a reproducibility and interface contract, not a
claim that observation scheduling dominates numerical error. The external
review's one-model experiment found the scheduling effect much smaller than the
`timestep=0.1` discretization error. Its interpolation experiment is encouraging,
but does not replace tests across models, distributions, and observables.

If incremental `advance!` is added later, either retain hidden integrator state
across calls or define a separate lattice-aligned stepping interface. Do not
impose that restriction on ordinary observation requests. Even a fixed-step
method need not have an exact arbitrary-time semigroup; test the specific
composition behavior the API promises rather than claiming it generally.

## 6. Store trajectories once

Store cell masses and requested times, derive density using grid weights, and
retain integration settings/diagnostics. Do not duplicate the trajectory as both
mass and density arrays.

```julia
path.times
mass(path, 1.0)       # shaped copy, if 1.0 was requested
density(path, 1.0)
final(path)           # StateDistribution; time remains path.times[end]
expectation(path, state -> state.k)
```

Prefer explicit time accessors. If indexing exists, integer indexing should be
positional; `at(path, t)` should mean lookup at a requested time. Specify tolerant
matching carefully rather than mapping arbitrary nearby times silently.

No raw mutable callback buffers in the everyday API. Optimized internal
observable evaluation can avoid copies without exposing cache lifetimes.

## 7. Economic extension contract: resources, not solver overrides

Candidate interface for supported one-capital models:

```julia
production(model, state)
resources(model, state)
marginal_product(model, state)
preferences(model)
state_schema(model)
default_grid_spec(model)
```

Define `resources` precisely: total income available **before capital
depreciation and consumption**. The shared resource equation is
`capital_drift = resources(model, state) - δ*k - consumption`.
For the downstream floor policy:

```julia
resources(model, state) = max(
    production(model, state) + model.transfer,
    model.income_floor,
)
```

One implementation supplies this income to the HJB solver, boundary validation,
policy evaluation, and residual diagnostics. It replaces downstream `update_v`
overrides. Output production remains distinct from disposable resources.
Any Euler-equation diagnostic must differentiate the actual resource rule when
appropriate; marginal production alone is insufficient under a binding floor.
The HJB itself need not require a differentiable resource rule at every point.

Support callable production objects such as `InterpProdFunction`, not just
hard-coded parametric formulas. Store their concrete callable type in model
parameters instead of requiring an abstract `Function` field. A model adapter
can call an existing `(k, z, ability)` object from a named state. Define behavior
at interpolation knots and outside observed support explicitly; do not invent
smooth derivatives at kinks.

A process provides dynamics on its axis. The operator layer embeds its generator
in the joint state layout. Population support/shares are economics; quadrature
and flattening are numerical implementation.

Before hiding `update_v`, demonstrate that the downstream income-floor model
and interpolated production can use the standard solver with no overrides.
A shared policy evaluator replaces the near-duplicate `SolvedModel` constructors.

## 8. Differentiation is a cross-cutting contract

ForwardDiff is used downstream, even though estimation is not today's primary
workflow. Preserve it deliberately rather than merely allowing Dual result arrays.

- **Fixed-grid derivatives:** holding realized coordinates fixed differentiates
  the model on that grid. This must work with Float64 coordinates and Dual model
  parameters. Grid checks must not reject a compatible scalar-type promotion.
- **Moving-grid derivatives:** constructing a grid inside a parameterized function
  differentiates coordinates and weights too. State this distinction explicitly;
  changing σ can change both the generator and the default grid.
- **Warm starts:** promote primal Float64 guesses into the solve's scalar type.
  Never inherit unrelated old tangents or mark convergence using only primal
  changes. A warm start is numerical initialization, not an economic dependency.
- **Regridding:** define which quantities are differentiated and how interpolation
  treats moving coordinates. Discrete shape/topology changes are not smooth.
- **Stationary distributions:** test derivatives through the generator, class
  weights, and grid measure where appropriate. A change in communicating-class
  topology or stationary uniqueness can make derivatives undefined.
- **Floors and interpolated production:** validate derivatives away from kinks;
  document piecewise behavior. Finite differences at a kink are not a smoothness
  guarantee. Do not silently strip Dual information or smooth the economics.
- Match AD against finite differences for consumption, transition moments, and
  a stationary moment. Include fixed-grid and moving-grid cases and warm starts.

## 9. Internal responsibilities and public surface

- Select tridiagonal/sparse/block solves internally, with numerical cross-checks.
- Replace sparse-Dual `Base.\` type piracy with an internal solve function.
- Keep factorization counters in instrumentation, not the stable public API.
- Retire variable-arity `update_v` only after the resources extension works.
- Move plotting into an extension. Remove unsupported two-asset exports.

Possible organization: `models/`, `processes/`, `grids.jl`, `operators.jl`,
`linear_solvers.jl`, `hjb.jl`, `solutions.jl`, `distributions.jl`, `evolution.jl`,
and `ext/`. This is a responsibility map, not a demand for many tiny modules.

## 10. Breaking release and downstream migration

No adapters or staged deprecations are needed. Rewrite examples/tests and give
a concise old-to-new usage guide. Keep numerical-default changes separate from
API changes so result differences remain attributable.

| Current entry point / extension | Candidate destination |
|---|---|
| `StateSpaceHyperParams` + `StateSpace` | `StateGrid(model; ...)` |
| `solve_HJB`, `solve_growth_model` | `solve(model, grid; ...)` → `GrowthSolution` |
| `Value`, `update_value_function!` | Result values and explicit copied warm-start guesses |
| `SolvedModel` | Policy evaluators on `GrowthSolution` |
| Ambiguous `iterate_g` overloads | Typed initial distributions plus `advance` / `trajectory` |
| `EvolutionWorkspace`, `workspace=` | Private state inside `Evolution` |
| `StateEvolution`, `save` modes | `trajectory`, `advance`, `observe` |
| `StationaryDistribution` | `stationary(fit, initial)` |
| `k_star` for Skiba | Already-exported `technology_crossing`; solved threshold remains separate |
| GrowthCounterfactuals `update_v` override | Economic `resources` rule including income floor |
| GrowthCounterfactuals `solve_HJB` override | Standard solve, explicit settings and warm starts |
| GrowthCounterfactuals `InterpProdFunction` | Concrete callable production adapter |

**Accepted scope: migrate GrowthCounterfactuals with the new API.** Use it as
the first downstream integration target and replace its solver overrides with
the shared resources contract and standard solve. Verify its resolved dependency
and actual method dispatch as part of that migration, then validate income-floor
counterfactuals and differentiation through the standard numerical core. This
is part of the agreed implementation scope; the present document remains a
design artifact, not a claim that migration has already occurred.

## First implementation target and acceptance tests

1. Explore stochastic RCK: solve, inspect policies/boundary diagnostics, evolve
   two initial distributions, and compare transition and stationary moments.
2. Solve an ability model with the downstream interpolated production and
   `max(production + transfer, floor)` resource rule. Use no solver overrides.
3. Warm-start a parameter change on a fixed grid; separately rebuild/regrid when
   changing a process parameter such as volatility moves the grid coordinates.
4. Differentiate a stationary moment with ForwardDiff and check finite
   differences away from economic kinks and class-topology changes.
5. Check shared-time invariance across observation requests, mass conservation,
   interpolation refinement, failure results, and rejection of unconverged fits.
6. Compare old/new numerical cores with identical explicit grids and settings;
   benchmark repeated solves and moment-only output without making timing a
   correctness assertion.

The main scope and behavior decisions are settled. Remaining specification work
covers axis/coordinate accessor details, numerical validation tolerances, the
explicit extrapolation option, and serialization of settings and initialization
snapshots. These can be resolved during implementation without another broad
user survey. Calibration-specific optimizers, data handling, and parameter
transforms can wait for a real exercise.
