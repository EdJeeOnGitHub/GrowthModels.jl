# Ability and individual productivity risk in discrete time

This extends [the deterministic design](DISCRETE_TIME_DESIGN.md). Ability scales
both technologies equally, productivity risk is individual, and inputs support
both an explicit finite Markov chain and a native log AR(1) approximation.
The work is based on the deterministic discrete-time branch; it does not change
the continuous-time process conventions or solver.

## Economics and timing

The current state is capital, productivity, and fixed ability where present.
Current productivity is observed before consumption and next-period capital are
chosen. Productivity then transitions; ability does not. Available resources are

```math
R(k,z,a)=azf(k),\qquad c+k'=R(k,z,a)+(1-\delta_d)k.
```

`ResourceModel` may add transfers and income floors, including nested rules.
Ability multiplies both Skiba technologies equally, leaving their production
crossing unchanged. Optimal investment switches can still depend on ability and
current productivity. Technology is selected each period without adoption history.

The Bellman equation is

```math
V(k,z,a)=\max_{k'}\left\{u(R(k,z,a)+(1-\delta_d)k-k')
 +\beta\sum_{z'}P_{zz'}V(k',z',a)\right\}.
```

A state-independent matrix `P` governs individual risk for all ability types.
Population shares affect aggregate outcomes, not individual policies. Aggregate
shocks, type-dependent processes, evolving ability, AD, and irreversible
investment are outside this implementation.

## API and state ownership

```julia
shock = MarkovChain(values=[0.8,1.2], transition=[0.9 0.1; 0.2 0.8])
# Or a process whose resolution is chosen by the grid:
shock = LogAR1(persistence=0.9, innovation_std=0.1, log_mean=0.0)
types = Population([0.8,1.2], [0.3,0.7])
model = DiscreteSkibaModel(productivity=shock, population=types)
grid = StateGrid(model; k=(n=200,bounds=(0.001,20.0)), z=(n=5,))
fit = solve(model,grid)
```

The two native model constructors accept optional `productivity` and `population`
keywords. There is no separate named model for each combination. Omitting an
argument removes its axis:

| Configuration | Axes |
| --- | --- |
| Neither | `(k,)` |
| Ability only | `(k, ability)` |
| Productivity only | `(k, z)` |
| Both | `(k, z, ability)` |

Capital is the fastest-changing dimension. Productivity and ability are categorical,
including the nodes of the AR(1) approximation. Queries and regridding cannot
interpolate between represented types or shock states. Population support belongs
to the model. Supplied Markov support belongs to the process; its resolution cannot
be changed by the grid. `transition(process(model), grid.z)` returns the
per-period row-stochastic matrix, without involving a generator.

Inputs and solution arrays retain the package's snapshot ownership rules.
Explicit capital coordinates remain fixed. Generated grids include the production
crossing and resource-floor crossings for all represented productivity/type pairs.
Default capital bounds use the highest represented productivity and ability,
including the actual requested AR(1) resolution. Bounds remain numerical proposals
that require sensitivity checks.

## AR(1) specification

`LogAR1` specifies the native per-period equation

```math
x_{t+1}=\mu+\rho(x_t-\mu)+\sigma\varepsilon_{t+1},\quad
\varepsilon\sim N(0,1),\quad z_t=\exp(x_t).
```

Here `log_mean=μ`, `persistence=ρ`, and `innovation_std=σ`. Require `abs(ρ)<1`
and `σ>=0`. The unconditional log variance is `σ²/(1-ρ²)`. No implicit level
normalization occurs: zero log mean does not mean unit expected productivity.

Use the symmetric Rouwenhorst recursion with `p=q=(1+ρ)/2` and evenly spaced log
nodes from `μ-σ*sqrt(n-1)/sqrt(1-ρ²)` to the corresponding upper endpoint. This
matches conditional and unconditional log means/variances and lag-one correlation
in exact arithmetic. It does not match every feature of the Gaussian process or
its nonlinear transformation. See [Kopecky and Suen (2010)](https://karenkopecky.net/RouwenhorstPaperFinal.pdf).

`StateGrid(...; z=(n=7,))` selects resolution; the default is seven states.
Positive volatility requires at least two states. Zero volatility collapses to
one state at `exp(μ)`, even when a larger `n` was requested. This represents the
stationary zero-volatility limit, not arbitrary off-mean deterministic AR(1)
transients. Explicit coordinates must exactly match the specified approximation.

## Numerical implementation

Compute expected continuation on the common capital grid before global search:

```math
W(k_j,z,a)=\sum_{z'}P_{zz'}V(k_j,z',a).
```

Expectation preserves piecewise linearity. The existing interval optimizer remains
a global optimizer of the interpolated problem, with the original utility and
budget-domain handling. Value iteration remains the reference algorithm.

Modified policy iteration uses a factorized action: first productivity expectation,
then interpolation at the fixed capital policy. It avoids repeatedly assembling the
joint matrix. A sparse joint transition is materialized once for the result and
for forward evolution. Its nonzero count is at most `2*Nk*Na*nnz(P)`. Larger dense
shock chains therefore have quadratic storage growth in the number of shock
states; matrix-free forward evolution remains a possible later optimization.

A solution privately caches expected continuation values. Off-grid queries and
individual simulations use that same snapshot, avoiding repeated AR(1)
construction and discrepancies between separate expectation calculations. Warm
starts use values only; a changed process always produces a new continuation cache.

## Distribution and individual paths

If the policy at `(k,z,a)` chooses capital between nodes `k_j` and `k_{j+1}`, the
joint transition assigns `P[z,z′]` times the two interpolation weights to those
capital nodes at each next productivity `z′`. Ability is unchanged. Consequently

```math
\mu_{t+1}=P_\pi^\top\mu_t.
```

This is the adjoint of the continuation operator used for valuation. It describes
individual risk averaged across a population, not a realized aggregate shock path.

`Evolution(fit)` returns a `DiscreteEvolution`; it accepts no `dt`. `advance`,
`trajectory`, and `observe` require nonnegative integer periods, preserve mass
without renormalization, and do not insert fractional-period transitions.
Adding observation dates does not change shared results. `advance` composes exactly
up to ordinary floating-point error.

`simulate(fit, initial_state, periods; rng)` returns a single individual's owned
states, consumption, and integer times. It chooses capital before drawing the next
productivity state, retains fixed ability, and re-optimizes at continuous capital.
There is no artificial capital lottery in this path. For `LogAR1`, simulations use
the fitted finite chain, not continuous Gaussian shocks. Passing an explicit RNG
makes results reproducible.

## Invariant measures and scientific interpretation

`stationary(fit, initial)` computes invariant probabilities with class weights
selected by the initial distribution and transient absorption. These are also the
long-run time-average probabilities. A periodic chain need not converge pointwise.
`require_aperiodic=true` rejects positive-mass periodic recurrent classes, even if
a specially chosen initial invariant measure itself would remain constant.

Without an initial distribution, population shares select independent ability
blocks only when each positive-share block has one closed class. Zero-share blocks
are skipped. Explicit initial distributions override population shares. A unique
invariant productivity distribution alone does not establish uniqueness of the
joint capital/productivity invariant distribution.

A stochastic policy switch is not automatically an absorbing poverty-trap
boundary. Evaluate target-hitting probabilities, transition times, invariant
mass, and conditional policies rather than assuming one scalar Skiba threshold.

Two approximation effects need separate treatment:

1. Linear capital interpolation splits population mass between capital nodes.
   Compare grid evolution against continuous-capital paths under the same shock
   chain and refine capital resolution and bounds.
2. Rouwenhorst support is bounded at any finite resolution; Gaussian shocks are
   unbounded. Rare escape probabilities and communicating-class topology can
   differ even when log moments match. Refine shock resolution and check support
   sensitivity before interpreting a numerical closed class as a permanent trap.

## Validation requirements

- One-state productivity and zero-volatility limits reproduce deterministic fits.
- Ability-only and identity-transition models agree with independently solved
  type-specific problems; changing population shares leaves policies unchanged.
- AR(1) conditional/unconditional log moments and stationary probabilities match
  their analytical counterparts, including negative and high persistence.
- Global stochastic choices dominate an independent dense action search.
- Factorized policy action agrees with the assembled sparse joint matrix and a
  fixed-policy linear-system solution.
- Evolution preserves mass and ability shares, composes in integer periods, and
  agrees with direct matrix powers and streaming observations.
- Invariant distributions handle absorption, multiple classes, zero-share types,
  and periodic examples with known answers.
- Off-grid simulation preserves budgets, timing, represented shock states, fixed
  ability, and RNG reproducibility.
- Capital/shock/bound refinement and Monte Carlo comparisons report sampling
  uncertainty and distinguish finite-chain results from Gaussian-process claims.
- The deterministic discrete-time and existing continuous-time suites continue
  to pass, including optional plotting and the minimum supported Julia version.

Reproducible examples are in `examples/discrete_risk.jl` and
`examples/discrete_risk_refinement.jl`; numerical results are recorded in
`DISCRETE_RISK_VALIDATION.md`.
