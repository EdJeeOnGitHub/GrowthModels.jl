# GrowthModels

One-capital growth models with continuous-time HJB and probability evolution,
plus native discrete-time Bellman solvers with individual productivity risk and
fixed ability types. For the version 1.0
breaking API revision, see
[MIGRATION.md](MIGRATION.md) for the old-to-new mapping.

Install the release directly from GitHub:

```julia-repl
julia> using Pkg
julia> Pkg.add(url="https://github.com/EdJeeOnGitHub/GrowthModels.jl", rev="v1.0.0")
```

Julia 1.9 or newer is required. Existing users should follow the migration guide
before updating: version 1.0 replaces the previous solver and state-space API.

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

## Native discrete time

Discrete-time models have per-period parameters and reversible investment:
`consumption + next_capital = resources + (1-depreciation)*capital`.
Capital cannot be negative. Skiba models select the best technology each period;
the capital offset is not an adoption payment.

```julia
discrete_model = DiscreteSkibaModel(discount=0.96, depreciation=0.08,
    risk_aversion=2.0, A_H=0.6, A_L=0.4, α=0.3, κ=2.0)
discrete_grid = StateGrid(discrete_model; k=(n=500, bounds=(0.001, 20.0)))
discrete_fit = solve(discrete_model, discrete_grid; history=true)
require_converged(discrete_fit)
next_k = next_capital(discrete_fit, (k=2.0,))
current_c = consumption(discrete_fit, (k=2.0,))
capital_path = [2.0]
for period in 1:100
    push!(capital_path, next_capital(discrete_fit, (k=last(capital_path),)))
end
```

`DiscreteRamseyCassKoopmansModel` uses the same preference keywords and `A`, `α`.
`preferences(model)` returns `(γ=..., discount=..., δ=...)`. `γ` and `δ` match the
continuous-time names, with `δ` per period here and a rate there. `discount` is a
per-period factor, not the rate `ρ`, and neither is converted to the other.
`risk_aversion=1` supports log utility in discrete time. These solvers use Float64;
automatic differentiation is not supported for discrete time. Native productivity
risk and population evolution are described below. Continuous-time APIs keep their
existing semantics. `ResourceModel` can apply transfers and income floors to a
discrete model, including nested resource rules.

The default modified policy iteration globally searches every piecewise-linear
continuation interval, allowing nonconcave values and policy jumps. Use
`algorithm=:value_iteration` as the reference method; `policy_steps` controls
fixed-policy sweeps in the accelerated method. Both value updates and the global
Bellman residual must converge. `maxiters=0` returns an inspectable failed fit.
Warm starts, explicit `regrid`, read-only snapshots, `plot_model`, and
`plot_diagnostics` are supported. `DiscreteGrowthSolution.transition` is a
row-stochastic policy matrix, not a continuous-time generator.

Off-grid policy queries re-optimize against the fitted continuation value instead
of interpolating policies across jumps. Exact ties choose lower next-period
capital. `diagnostics.alternative_capital` and `choice_gap` describe another
local maximum when one exists (`NaN`/`Inf` otherwise); small gaps indicate
sensitivity, not a uniquely identified threshold.

Generated discrete grids insert known production and resource-floor crossings,
so their final length can exceed `n`. Explicit coordinates remain unchanged.
Numerical bounds constrain capital choices: a positive `kmin` is an artificial
saving constraint, and a low `kmax` can suppress investment. Inspect
`diagnostics.boundary.lower_choice` and `upper_choice`, and refine bounds and
resolution together. Zero is allowed on discrete grids, but zero income at zero
capital has negative-infinite value for log utility or risk aversion above one;
the solver reports this unsupported finite-value domain instead of adding a
consumption floor. No continuation extrapolation occurs during solving.

`diagnostics.value_error_bound` bounds iteration error for the interpolated
problem only. Thresholds and policies require separate refinement checks. See
[the capital-path example](examples/discrete_time.jl),
[the reproducible refinement experiment](examples/discrete_refinement.jl), and
[the design document](DISCRETE_TIME_DESIGN.md).

## Discrete-time ability and productivity risk

Both discrete model constructors accept `population=Population(...)` and
`productivity=MarkovChain(...)` or `LogAR1(...)`. Ability and productivity multiply
both production technologies equally. Ability is fixed; each individual's current
productivity is observed before choosing capital, and its next value is drawn
subsequently. Population shares change aggregate results without changing policies.

```julia
risk = LogAR1(persistence=0.9, innovation_std=0.1, log_mean=0.0)
types = Population([0.8, 1.2], [0.3, 0.7])
risky_model = DiscreteSkibaModel(productivity=risk, population=types)
risky_grid = StateGrid(risky_model; k=(n=150, bounds=(0.001, 20.0)), z=(n=5,))
risky_fit = solve(risky_model, risky_grid)
require_converged(risky_fit)

initial_masses = zeros(size(risky_grid))
initial_masses[50, 3, :] .= types.shares
risky_initial = distribution(risky_grid; masses=initial_masses)
risky_evolution = Evolution(risky_fit) # Integer periods; no dt.
risky_path = trajectory(risky_evolution, risky_initial, 0:20)
risky_stationary = stationary(risky_fit, risky_initial)

using Random
individual = simulate(risky_fit,
    (k=2.0, z=risky_grid.z[3], ability=0.8), 20; rng=MersenneTwister(42))
```

For an explicit chain, use
`MarkovChain(values=[0.8,1.2], transition=[0.9 0.1; 0.2 0.8])`.
Rows are conditional on current productivity; entries are next-period
probabilities. Inputs are validated without normalization. This convention is
independent of continuous-time OU/Poisson economic conventions.

`LogAR1` models `log(z′) = log_mean + persistence*(log(z)-log_mean) + ε`,
where `innovation_std` is the **innovation** standard deviation. Rouwenhorst
approximation uses `z=(n=...)`, defaulting to seven states. Zero volatility
collapses to one stationary state. No level normalization occurs: `log_mean=0`
does not imply mean productivity one. Productivity and ability are categorical;
queries must use represented values, and regridding cannot interpolate across
shock states or types. Ability-only models have axes `(k, ability)` without a dummy
productivity axis. Models with both have `(k, z, ability)`.

`transition(process(risky_model), risky_grid.z)` returns the exogenous matrix.
`risky_fit.transition` is the joint capital/productivity/type matrix, acting on
vectorized values. Fixed-policy sweeps use a factorized operator, while the sparse
joint matrix is stored once; its storage can grow quadratically in the number of
shock states for dense chains. Off-grid queries and `simulate` share cached
expected continuation values. Simulations keep capital continuous and sample the
finite productivity chain, rather than adding capital-grid lotteries.

Discrete `stationary` returns invariant/time-average probabilities, including
initial-dependent absorption weights. Periodic chains need not converge at each
date; `require_aperiodic=true` rejects positive-mass periodic classes. Without an
initial distribution, population shares select ability blocks only when each
positive-share block has a unique closed class. `advance`, `trajectory`, and
`observe` require integer periods and preserve ability shares and total mass.
These describe individual risk, not realized paths under common aggregate shocks.

Refine capital bounds/resolution and shock resolution separately. Bounded
Rouwenhorst support can change rare escape probabilities and apparent absorbing
classes relative to unbounded Gaussian shocks. A small Bellman residual or matched
log moments does not settle those questions. See the
[stochastic design](DISCRETE_RISK_DESIGN.md),
[validation evidence](DISCRETE_RISK_VALIDATION.md), and
[refinement experiment](examples/discrete_risk_refinement.jl).

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

`solve` returns a consistent result structure within each time formulation. Check `fit.converged`,
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

Continuous-time convergence checks both value updates and the nonlinear HJB residual, including
Dual tangents. Default `Delta` is 1 for γ<1 and 1000 otherwise; resolved settings
are recorded. γ=1 (log utility) is unsupported by the continuous-time solver.

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

A custom continuous-time one-capital model supplies `production(model, state)`,
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
