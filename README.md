# GrowthModels

[![Build Status](https://github.com/EdJeeOnGitHub/GrowthModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EdJeeOnGitHub/GrowthModels.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Numerical conventions

The HJB generator `A` acts on value functions. Its transpose `A'` advances
**cell probabilities**, not densities. `iterate_g(g, A_t, dx_stacked)` accepts a
density and cell weights, evolves `dx_stacked .* g`, and converts back. It
preserves the input weighted mass without renormalizing. The two-argument
`iterate_g(p, A_t)` instead accepts cell probabilities. `StateEvolution`
normalizes the initial density once.

`StationaryDistribution(A_t, dx_stacked; initial_density = g)` computes the
limiting density associated with `g`, including absorption from transient states.
For disconnected ability groups, their initial probability shares are preserved.
If several closed communicating classes exist, `initial_density` is required;
`StateEvolution(...; stationary=true)` supplies it automatically. With a unique closed class, the
keyword can be omitted.

HJB policy selection compares Hamiltonians when both drift directions are
admissible, choosing one policy for both consumption and the generator. The OU
generator upwinds its drift according to its sign, keeping off-diagonal rates
nonnegative. This changes coarse-grid results relative to the historical MATLAB
RBC fixture, which uses a forward-only drift stencil. That fixture remains a
legacy comparison in the tests; the corrected solver is checked separately.

## Repeated solves and warm starts

```julia
m = RamseyCassKoopmansModel()
hp = StateSpaceHyperParams(m; Nk = 200)
first_fit = solve_HJB(m, hp; verbose = false)
next_model = RamseyCassKoopmansModel(A = m.A * 1.01)
next_fit = solve_HJB(next_model, hp;
    init_value = first_fit.value, crit = 1e-8, verbose = false)
```

`init_value` accepts a `Value` or an array of the grid's shape. It is copied and
honored, including an all-zero guess. Omit it (or pass `nothing`) to use the
model's default guess. Earlier results and their diagnostics are never mutated.
`maxit=0` returns the supplied/default guess without iteration. Convergence
requires both a small value update and a nonlinear HJB residual below `crit`;
policies using a floored marginal value cannot certify convergence. The default
pseudo-time step is `Delta=1` for γ<1 and `Delta=1000` otherwise. An explicit
`Delta` overrides this choice. γ=1 (log utility) is currently unsupported and
raises an explicit error. Constructing `SolvedModel` from a failed solve also
raises an explanatory error. Convergence
history contains one initial `Inf` and one distance per update. With dual
numbers, convergence also requires their tangents to converge.

Each solve allocates its geometry, policy arrays, and sparse structure once.
The default `linear_solver=:auto` uses tridiagonal elimination for deterministic
models and separate sparse blocks for fixed-ability groups. Sparse numerical
factors are updated each iteration with a fixed symbolic pattern. A generator
that couples ability groups automatically uses a full sparse solve. Use
`linear_solver=:sparse` to cross-check against one full sparse system.
Differentiated sparse solves share a primal factorization across all tangent
right-hand sides, including nested dual numbers.

## Cached forward evolution and output control

```julia
sm = SolvedModel(next_model, next_fit)
g0 = ones(length(next_fit.value.v))
times = collect(0.0:1.0:20.0)

# Four internal steps per unit of time, but save only the requested times.
path = StateEvolution(g0, sm, times; dt = 0.25)
final = StateEvolution(g0, sm, times; dt = 0.25, save = :final)

# Stream moments without retaining distribution snapshots. The callback receives
# cell probabilities, so no extra cell-width weighting is needed.
mean_capital = Float64[]
streamed = StateEvolution(g0, sm, times; dt = 0.25, save = :none,
    callback = (t, mass) -> push!(mean_capital, dot(vec(sm.variables.k), mass)))

stationary = StationaryDistribution(path) # separate, explicit computation
```

The `dot` example requires `using LinearAlgebra`. `save=:all` (the default)
stores every requested observation, `:final` stores only the last, and `:none`
stores no trajectory. Every result has `final_density` and `final_time`.
Callbacks run at observation times, including the initial time, independently of
storage mode. The callback's mass buffer is borrowed: do not mutate it or retain
it without copying. `dt` is a maximum internal step; each observation interval
is divided into equal substeps that land exactly on its endpoint. The default
for real observation times is `dt=0.1`, independent of observation spacing.
Explicit `dt=nothing` restores one step per observation interval. Adding times
that do not align with the internal grid can still change the subdivision;
check time-step refinement for the accuracy your application needs.

Stationary computation is **no longer automatic**: `stationary_dist` is
`nothing` unless `stationary=true` is requested. You can instead call
`StationaryDistribution(path)` or
`StationaryDistribution(sm; initial_density=g0)` afterward. The integer overload
`StateEvolution(g0, sm, T; implicit_steps=4)` retains its existing convention of
T observations at `0:T-1`. Real-valued observation times are supported, and
`path[t]` retrieves a saved density.

For repeated simulations with the same grid, a workspace can be reused:

```julia
using SparseArrays
weights = create_dx_stacked(sm.variables.k[:, 1], 1) # deterministic example
A_t = sparse(sm.value.A')
workspace = EvolutionWorkspace(A_t, weights; time_step = 0.25)
path = StateEvolution(g0, sm, times; dt = 0.25, workspace = workspace)

# Low-level single step; no automatic normalization.
g1 = iterate_g(g0, A_t, weights; time_step = 0.25, workspace = workspace)
```

The workspace owns snapshots of its generator and weights, plus reusable solve
buffers. Passing `workspace` to `StateEvolution` or the matrix-taking
`iterate_g` methods refreshes it when the generator or timestep changes.
Alternatively call `update_evolution!(workspace, new_A_t; time_step=0.5)` and
then `iterate_g!(g, workspace)`. Changes to values reuse symbolic analysis;
changes to sparsity rebuild it. Cell-weight changes alone need no factorization.
Changing the number of states requires a new workspace. Workspace fields are
implementation details and must not be mutated directly. Each concurrent
simulation needs its own workspace.

For fixed-ability models, `StateEvolution` automatically factors independent
capital/productivity blocks. A manually constructed workspace can request this
with `blocksize=Nk*Nz`; cross-block transitions are rejected. The initial density
is normalized once by default; pass `normalize=false` to preserve a nonunit input
mass. No renormalization is applied during time stepping. When requested, the
stationary density in an evolution result (including `StationaryDistribution(path)`)
has the same total mass as that result. The matrix-taking stationary function
continues to return a unit-mass probability density.

## Validation and benchmarks

Run the full suite with:

```sh
GKSwstype=100 julia --project=. test/runtests.jl
```

`test/test-performance-contracts.jl` checks cache invalidation, selective storage,
streamed moments, warm starts, fixed sparse patterns, block/tridiagonal
agreement, and finite-difference checks of derivatives. The earlier P1 tests
continue to cover generator positivity and probability conservation.

Run reproducible performance comparisons with:

```sh
julia --project=. test/benchmark-performance.jl
```

The benchmark reports medians of three runs after compilation, including setup
and factorization costs. It compares cached/uncached forward steps, full/final
output storage, cold/same-model warm HJB solves, and sparse differentiation.
An optional first argument loads another checkout using the current project's
dependencies for an HJB/differentiation comparison. Timing is informational;
correctness and reuse contracts, rather than machine-dependent timing limits,
are enforced in tests.

## Scientific defaults and remaining limitations

Default OU productivity bounds are `exp(±z_width * σ / sqrt(2θ))`, with
`z_width=4`. This represents a zero-mean OU process in log productivity.
Parametric stochastic models size the upper capital bound from the highest
productivity and ability, then apply `kmax_f`. Custom nonparametric production
functions still require the caller to assess their chosen capital bounds.
Wider grids at a fixed number of points have coarser resolution: refine both
bounds and grid size when validating numerical results.

`technology_crossing(model)` names the crossing of the two Skiba production
branches. The historical `k_star` alias does **not** compute the optimal Skiba
threshold. SmoothSkiba steady-state helpers remain asymptotic approximations;
its high approximation no longer adds a capital shift absent from production.
Stochastic steady-state helpers are reference values, not claims about the
stationary distribution of optimal capital under uncertainty.

State constraints are checked across **every lower-capital boundary state**.
Negative net output at high capital is allowed when capital can drift inward;
it is not, by itself, evidence of an invalid grid. Boundary derivatives use the
same floored zero-drift marginal value convention throughout the solver.

`flat_transfer` remains a solver option in this compatibility-preserving release,
but is retained with the solved policy and included in interpolated capital
drift. The unfinished two-asset model remains experimental and unsupported.
The larger grid/result/evolution API redesign and further performance tuning
are deferred. See [CORRECTNESS_REVIEW.md](CORRECTNESS_REVIEW.md) for validation
and changes that require regenerating default-based results.
