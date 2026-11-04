# Breaking API migration

There are no compatibility adapters. Economic parameters and resource rules
belong to models; numerical settings belong to grid construction and `solve`.

| Previous API | Replacement |
|---|---|
| `StateSpaceHyperParams`, `StateSpace` | `StateGrid(model; k=..., z=...)` |
| `solve_HJB`, `solve_growth_model` | `solve(model, grid)` |
| `Value`, `init_value` | `fit.value`, `initial=fit` or a shaped array |
| `SolvedModel` | `GrowthSolution` returned by `solve` |
| `sm.variables.c`, `sm.value.v` | `fit.policy.consumption`, `fit.value` |
| `sm.variables.k` | `coordinates(fit.grid).k` |
| `flat_transfer` solver keyword | `ResourceModel(base; transfer=...)` |
| `update_v` overrides | `resources(model, state)` or `ResourceModel` |
| `EvolutionWorkspace`, `workspace=` | `Evolution(fit; dt=...)` |
| `StateEvolution`, `save=:all` | `trajectory(evolution, initial, times)` |
| `save=:final` | `advance(evolution, initial, duration)` |
| `save=:none`, callbacks | `observe(evolution, initial, times, named_observables)` |
| `iterate_g` | Typed distribution plus `advance` / `trajectory` |
| `StationaryDistribution` | `stationary(fit, initial)` |
| `k_star` | `technology_crossing` (not the optimal threshold) |

Provide unit-mass probability arrays or density arrays with unit weighted
integral. Normalize explicitly before construction if needed. Density is per
unit capital and mass over productivity/ability states. Accessors return shaped
copies; do not depend on flattened storage or mutate solver caches.

`dt` is required. Observation interpolation replaces endpoint-dependent
substeps, so off-lattice output can differ from the previous implementation.
Shared observations are now independent of the requested schedule. Stationary
calculation is separate and has no timestep. Incremental stepping is deferred.

`fit.settings` records the resolved solve options and `fit.initial_value` owns
the actual starting array. A failed fit retains finite values and policy with
`converged=false`; downstream economic calculations require convergence.
Warm starts with changed coordinates require explicit `regrid`.

For GrowthCounterfactuals, replace solver overrides with the standard
`GrowthModels.solve` call and a resource model implementing the income floor.
Use `StateGrid`, model-owned population support/shares, and `fit.value` for value
interpolation. Qualify `GrowthModels.solve` where packages such as Roots also
export `solve`. Verify the active environment resolves this checkout: changing
source files alone does not update an environment pinned to an old git tree.

## Downstream grid defaults

GrowthCounterfactuals retains its split capital-grid formula and empirical
population bin shares. Productivity bounds now use the analytical central 98%
interval of the stationary OU law instead of quantiles from 100 simulated
paths. This eliminates grid construction's global RNG reset, but changes its
realized bounds and can change counterfactual numbers. Treat that as a separate
numerical-default change: supply a grid with the previous realized coordinates
when comparing numerical cores, and regenerate default-grid analyses.

The checked-in `test/integration/Project.toml` resolves sibling working checkouts
for the migrated package. The broader aggregate-transfers root and
joint-generative-model and filter environments are migrated and currently
pin local Git revision `a6230e7`. These pins require the local GrowthModels Git
repository; replace them with the published revision when releasing the change.
Active notebook API calls are migrated; see `API_COMPLETION.md` for pre-existing
exploratory-cell limitations. The historical loader now
keeps archived source unchanged and substitutes the shared recovery numerical
adapter for its retired solver/generator methods.

Saved Julia objects containing retired `SolvedModel` or `StateSpace` types must
be regenerated with the migrated callers. Keep original archives and their
environment pins for historical reproduction; do not overwrite them with new
results that use different numerical defaults.

The analytical Skiba export and traps notebook now share
`code/growth-models/TrapProbabilities.jl` downstream. They initialize explicit
uniform cell mass and use `Evolution(fit; dt=1)` for twelve time units. Their
forward and backward threshold events both mean `k >= threshold`; previously
the backward event used strict `>`. This changes results only at exact threshold
nodes. Per-ability output retains joint probabilities, rather than silently
changing them to conditional probabilities.

Continuous-shock recovery simulations previously used multilinear interpolation
and linear extrapolation on all three axes. They now opt into
`AnalysisPolicy.continuous_ability_policy(fit; extrapolate=true)` downstream to
retain that approximation. Bounded-grid simulations reject extrapolation.
This is an explicit analysis assumption, not an extension of the core solver's
categorical ability semantics or evidence of an optimal policy outside the grid.
The analysis environment declares its own Interpolations dependency.

## Compositional resource adjustments

`ResourceModel(base; transfer, income_floor)` now computes
`max(resources(base, state) + transfer, income_floor)`. Previously it started
from `production(base, state)`, discarding custom resource rules and inner
wrappers. Built-in models with unadjusted resources behave as before. Nested
wrappers now apply in order; use the unwrapped production model explicitly when
replacement rather than composition is intended.

## Deep-module refactor

- Model extensions use `process(model)` rather than a required process field.
  `ResourceModel` no longer forwards arbitrary fields. Use `preferences`,
  `production`, `resources`, `process`, and `population` on wrapped models.
- Process extensions implement `process_axis(process, specification)` as well as
  `generator`. Axis resolution returns coordinates and either `:interpolable`
  or `:categorical`; the process validates its own support and option keys.
- Unknown grid options throw. Custom OU defaults with explicit bounds no longer
  require a `width` field.
- Array inspection is read-only. Replace mutations of result/grid properties
  with explicit `copy` calls; pass editable values as a new solve's `initial`.
  Evolution factorization storage is not a public property.
- `stationary(fit)` uses population shares if each positive-share ability block
  has one closed class. An explicit initial distribution always takes precedence.
  Multiple classes within a positive-share block still require explicit input;
  zero-share ambiguous blocks have zero mass.
- The internal `StateSpace`, `HyperParams`, `StateSpaceHyperParams`, and duplicate
  diffusion assembly paths are removed. Use `StateGrid` and axis `generator`.
- `Interpolations` is no longer a GrowthModels dependency. Applications that use
  it directly must declare it themselves.

## Separate numerical-default change: capital support

For built-in parametric stochastic models, omitted capital bounds now account
for the *resolved* productivity coordinates and population ability support.
Changing `z.width`, supplying productivity coordinates, or changing ability
support may therefore change the default capital grid. Previously productivity
width four informed the capital proposal even when another support was selected.
Explicit capital bounds and fixed grids remain unchanged. Re-run default-grid
analyses and assess bounds and resolution together; the strict explicit-grid
refactor comparison does not measure this intentional default change.

## Separate numerical fix: residual evaluation on stretched grids

The nonlinear HJB residual now evaluates generator action as weighted value
differences, using the zero-row-sum identity, instead of subtracting large
level products. Convergence tolerances are unchanged. A 900-point RCK grid with
power 3.9 previously reported a residual around `1.31e-6` after 1000 iterations,
while a 256-bit recomputation of the same iterate gave `5.53e-7`. The stable
calculation agrees with high precision and avoids that false failure. Very
ill-conditioned grids can still fail legitimately; this is not a relaxation of
the acceptance criterion.


The structural panel notebook now samples from cell masses rather than density
values, which previously over-weighted narrow cells on stretched grids. Its
integer horizon retains the historical `0:T-1` clock. KKT analysis derivatives
are reconstructed from public value arrays and state-constraint boundary values;
private workspace access is no longer required. These downstream numerical
interpretations are explicit and do not change the core refactor baseline.
