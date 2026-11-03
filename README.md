# GrowthModels

Continuous-time, one-capital growth models solved with a finite-difference HJB
and probability evolution. This is a breaking API revision; see
[MIGRATION.md](MIGRATION.md) for the old-to-new mapping.

## Solve and simulate

```julia
using GrowthModels

model = RamseyCassKoopmansModel()
grid = StateGrid(model; k=(n=300,))
fit = solve(model, grid)
require_converged(fit)
c = consumption(fit, (k=grid.k[100],))

p = fill(1 / length(grid), size(grid))
initial = distribution(grid; masses=p)
evolution = Evolution(fit; dt=0.1)
path = trajectory(evolution, initial, [0.0, 0.25, 1.0, 5.0])
mean_k = expectation(path, state -> state.k)
last_distribution = advance(evolution, initial, 5.0)
moments = observe(evolution, initial, [0.0, 1.0, 5.0],
    (mean_k=state -> state.k, second_k=state -> state.k^2))
long_run = stationary(fit, initial)
```

`trajectory` stores cell probabilities once. `mass(path, 0.25)` and
`density(path, 0.25)` return shaped copies. `at(path, t)` looks up a requested
time; unrequested times raise an error. Integer indexing is positional.
`final(path)` returns a distribution, with time retained in `path.times[end]`.

Observation times may be arbitrary. Integration uses implicit Euler on a fixed
lattice anchored at `t0` (default zero), with linear interpolation of bracketing
probabilities. Adding observations does not change results at shared times.
Refine `dt` to assess accuracy. `advance` is a one-shot duration request, not
an incremental integrator with an arbitrary-time composition guarantee.
Reuse `Evolution` for another initial distribution; use separate objects for
concurrent simulations. Construct a new evolution for a changed fit.

## Economics and grids

Production, resource rules, processes, and population support belong to the
model. Grid resolution belongs to `StateGrid`. Canonical axes are capital,
productivity, and ability, where present; keyword order never changes layout.
`grid.k`, `grid.z`, and `grid.ability` return read-only coordinate views.
`coordinates(grid)` returns shaped named coordinate arrays.

```julia
base = StochasticSkibaAbilityModel()
model = ResourceModel(base; transfer=0.1, income_floor=0.3,
    population=Population([0.8, 1.2], [0.3, 0.7]))
grid = StateGrid(model; k=(n=300,), z=(n=40, width=4))
fit = solve(model, grid)
```

`resources(model, state)` is income before depreciation and consumption:
`capital_drift = resources - δ*k - consumption`. `ResourceModel` applies
`max(resources(base, state) + transfer, income_floor)`. The solver, residual diagnostics,
and policy evaluation use the same rule. `FunctionalModel` supports concrete
callable production objects with arguments `(k, z, ability)`; economic
customizations should not override solver internals.

Explicitly supplied coordinates stay fixed when model parameters change.
Reconstruct a grid to change its coordinates. Default OU bounds are
`exp(±width*σ/sqrt(2θ))`, with `width=4`; ability support and population shares
belong to the model. Default parametric capital bounds account for highest
productivity and ability. Custom production requires explicit assessment of
capital bounds. Wider bounds at unchanged resolution can increase grid error.
Inspect boundary drift and `boundary_mass(distribution)` and refine bounds and
resolution together.

## Probability contract

Inputs must be nonnegative, finite, and have total probability one within
`1e-10`. There is no automatic normalization. For density input, validation
uses the weighted integral:

```julia
raw_density = ones(size(grid))
pdf = raw_density ./ sum(raw_density .* cell_weights(grid))
initial = distribution(grid; density=pdf)
```

Density is per unit capital and mass over Markov states, including discretized
OU states. Arrays have grid shape, e.g. `Nk × Nz × Nability`. `mass` and
`density` return copies. Distributions have no time; trajectories own times.
Evolution checks conservation without renormalization. `stationary(fit,
initial)` preserves limiting class shares, including transient absorption.
Without an initial distribution, population shares select the ability-block
weights if each positive-share type has a unique closed class. Otherwise an
initial distribution is required. Zero-share types contribute no mass. Explicit
initial distributions always override these population defaults.
Stationary calculation has no timestep.

## Failure, warm starts, and differentiation

`solve` always returns the same result structure. Check `fit.converged`,
`fit.diagnostics`, `fit.settings`, and the copied `fit.initial_value`.
Iteration-limit or numerical-breakdown failures retain the last finite value
and its policy. Invalid input, or an initial guess with no finite policy, throws.
Policy evaluation, evolution, and stationary calculation reject failed fits;
raw arrays remain inspectable. `require_converged(fit)` provides an explicit
check for batch jobs.

```julia
model = RamseyCassKoopmansModel()
grid = StateGrid(model; k=(n=300,))
fit = solve(model, grid)
next_model = RamseyCassKoopmansModel(A=model.A * 1.01)
next_fit = solve(next_model, grid; initial=fit)
# When coordinates change:
# guess = regrid(fit, next_grid)
# next_fit = solve(next_model, next_grid; initial=guess)
```

Initial guesses are copied. `regrid` rejects
extrapolation by default; use `extrapolate=true` explicitly if appropriate.
Discrete population support cannot be changed by continuous interpolation.

Convergence checks both value updates and the nonlinear HJB residual, including
Dual tangents. Default `Delta` is 1 for γ<1 and 1000 otherwise; resolved settings
are recorded. γ=1 (log utility) is currently unsupported.

ForwardDiff differentiates fixed-grid models while holding coordinates fixed.
Constructing a grid inside the differentiated function also differentiates its
coordinates and weights. Warm starts and regridded guesses are initialization:
old tangents are discarded. Floors, interpolation knots, policy switches, and
changes in communicating-class topology can make derivatives nonsmooth.
Sparse differentiation uses internal factorization helpers without extending
`Base.\` for external types.

## Plotting and validation

Install `Plots` in your application environment and load it to activate `plot_model(fit)` and
`plot_diagnostics(solve(model, grid; history=true))`. The numerical package does
not load Plots itself.

```sh
GKSwstype=100 julia --project=. -e 'using Pkg; Pkg.test()'
```

The API tests cover scientific invariants, finite-difference checks of AD,
probability contracts, observation invariance, and numerical agreement with the
pre-redesign solver on identical explicit grids. Implementation and downstream
validation progress is tracked in [API_IMPLEMENTATION.md](API_IMPLEMENTATION.md).

`technology_crossing` is the crossing of production technologies, not the
optimal Skiba threshold. SmoothSkiba steady-state helpers are asymptotic
references. Stochastic steady states are reference values, not stationary
capital moments. The unfinished two-asset model is not part of this API.

## Extending models and processes

A custom one-capital model supplies `production(model, state)`,
`preferences(model)` returning `(γ=..., ρ=..., δ=...)`, and
`default_grid_spec(model)`. Override `resources` for disposable-income rules;
the default is production. Override `population` for ability support/shares.
Supported canonical schemas are `(:k,)`, `(:k,:z)`, and
`(:k,:z,:ability)`; other schemas raise an error. Stochastic models implement `process(model)`; deterministic models return
`nothing`. Wrappers delegate this interface rather than forwarding arbitrary
fields. `marginal_product` defaults to differentiating
production with respect to capital.

A custom process implements `process_axis(process, specification)`, returning
`(coordinates=..., kind=:interpolable)` or `kind=:categorical`, and
`generator(process, coordinates)`, returning the
axis generator acting on values. Rows sum to zero and off-diagonal rates must
be nonnegative. The operator layer validates and embeds that generator in the
joint grid; it owns stacking and independent ability blocks. Supply explicit
productivity coordinates or bounds for a custom process. This interface covers
one exogenous productivity process with fixed ability classes.

## Snapshot ownership

Solution values, policies, generators, diagnostic arrays, grid axes, population
arrays, and trajectory storage are read-only inspection views. Indexing does not
copy an entire array. Writes (including through views or broadcasting) throw;
use `copy(fit.value)` for editable data. `mass` and `density` still return shaped
copies. `fit.model` returns a detached model snapshot. Evolution factors and
workspace buffers are private. Deliberate access to internal fields is outside
the supported interface.

Grid option typos raise errors; for example `k=(n=50, bound=(0.1,5.0))` must use
`bounds`. Explicit coordinates bypass unused model defaults. Categorical support
cannot be changed by regridding or queried between represented types.
