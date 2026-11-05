# Native discrete-time growth models

Status: deterministic milestones implemented on the discrete-time feature branch.
See [implementation and validation evidence](DISCRETE_TIME_VALIDATION.md).
Ability, individual productivity risk, and distribution evolution are implemented
by [the follow-up design](DISCRETE_RISK_DESIGN.md). Other future work remains
deferred as specified below.

## Purpose and agreed scope

Add native discrete-time growth models to GrowthModels.jl in this repository
and package. The first milestone supports deterministic Ramsey and Skiba models,
including nonconcave production, kinked value functions, and discontinuous
policies. Preserve the existing continuous-time API and release behavior.

Decisions agreed with the user:

- Native discrete-time economics, rather than a time discretization of the HJB.
- Reversible investment: agents can consume undepreciated capital.
- Choose the best available production technology each period, without adoption
  history or a one-time adoption payment.
- Deterministic models are sufficient for the first version.
- Automatic differentiation is outside the initial scope.
- Dedicated discrete-time constructors, with shared economic implementation
  underneath, rather than wrappers around continuous-time models.

No borrowing is the proposed baseline: next-period physical capital is
nonnegative. Stochastic productivity, ability heterogeneity, distribution
evolution, irreversible investment, and continuous-time conversion are deferred.

## Design rationale

The interface should describe one economy without requiring knowledge of
parameter precedence or numerical internals. A wrapper around an existing
`SkibaModel` would inherit continuous-time `ρ` and `δ`, then require rules about
which fields are ignored, converted, or overridden. Dedicated constructors avoid
that ambiguity.

This applies Ousterhout's information-hiding and deep-module principles: move
complexity into the implementation and keep the caller's conceptual burden
small. It is a design judgment for this package, not a prescription from the
author. See [Ousterhout's abstraction notes](https://web.stanford.edu/~ouster/CS349W/lectures/abstraction.html).

Dedicated constructors must not lead to duplicated solvers. Ramsey and Skiba
should use the same Bellman implementation through shared economic interfaces.
Reuse production formulas, grid containers, ownership conventions, and useful
validation machinery where their contracts fit. Keep HJB and Bellman numerical
invariants separate.

Avoid building a general dynamic-programming framework before these two concrete
models work. Extract additional abstractions when their benefit is demonstrated.
Time-independent technology objects may eventually help custom model composition,
but are not required in the first public interface.

## Economic contract

Capital `k` is measured at the beginning of a period. Current production is
available for consumption or investment. Next-period capital is chosen after
observing the current state. For the deterministic model, define available goods
as

```math
B(k) = R(k) + (1-\delta_d)k,
```

where `R` is disposable production income before depreciation and consumption.
This preserves the existing meaning of `resources`.

The budget and admissible choices are

```math
c + k' = B(k), \qquad k' \geq 0, \qquad c \geq 0,
```

with strictly positive consumption wherever required for finite utility. Since
investment is reversible, there is no restriction `k' >= (1-δ_d)k`.

The Bellman equation is

```math
V(k) = \max_{k' \in \Gamma(k)}
\left\{ u(B(k)-k') + \beta V(k') \right\},
\qquad 0 < \beta < 1.
```

Both discounting and depreciation are per period. A native model does not infer
either from continuous-time rates. Validate `0 <= depreciation <= 1` and positive
risk aversion. The period's calendar interpretation belongs in user calibration;
the first solver does not need a numerical timestep.

Use the existing CRRA convention for risk aversion other than one:

```math
u(c)=\frac{c^{1-\gamma}}{1-\gamma}.
```

Proposed addition: support log utility explicitly at `γ = 1` in the discrete-time
implementation. This must not change the existing continuous-time validation or
promise differentiability of value levels across utility normalizations.

At zero consumption, utility is negative infinity for log utility and CRRA with
`γ > 1`; for `0 < γ < 1`, its continuous extension is zero. Do not silently add
an economically meaningful consumption floor to make calculations finite.
States without a finite feasible solution must produce an informative error or
explicit unsupported-domain diagnostic. In particular, zero capital with zero
income may have negative-infinite value for log utility or `γ > 1`.

### Technology interpretation

Retain the existing Skiba production envelope:

```math
f(k) = \max\{A_L k^\alpha, A_H(k-\kappa)_+^\alpha\}.
```

The parameter `κ` is a capital offset in the high-production branch. It is not a
one-time adoption expenditure. Technology can be selected anew each period.

The production crossing is not the optimal Skiba threshold. The latter reflects
indifference between distinct future investment paths and may move with
preferences and depreciation even when the production crossing is unchanged.

## Proposed public interface

```julia
model = DiscreteSkibaModel(
    discount=0.96,
    depreciation=0.08,
    risk_aversion=2.0,
    A_H=0.6,
    A_L=0.4,
    α=0.3,
    κ=2.0,
)

grid = StateGrid(model; k=(n=500, bounds=(0.001, 20.0)))
fit = solve(model, grid)
require_converged(fit)

c = consumption(fit, (k=2.0,))
kp = next_capital(fit, (k=2.0,))
```

Provide a corresponding `DiscreteRamseyCassKoopmansModel`. Final naming should
be reviewed for consistency with the package before implementation.

The default algorithm must correctly handle the advertised nonconcave models.
Users should not have to select a special solver to avoid local optima. Advanced
algorithm controls may be added without making them part of ordinary usage.

Discrete-time model parameters have one owner. Do not reuse the production
smoothing parameter named `β` in `SmoothSkibaModel` as an implicit discount factor;
use `discount` in the new interface.

Reuse `solve`, `consumption`, `regrid`, and `require_converged` where semantics
agree. Introduce `next_capital` for discrete time. Preserve the continuous-time
meaning of `capital_drift`.

Write constructor documentation and a complete solve-and-capital-path example
before committing to the internal type layout. The example may initially iterate
`next_capital` directly; distribution trajectories are not part of milestone one.

## Implementation boundaries

Use separate discrete-time model, Bellman workspace, and solution types. Proposed
source organization is `DiscreteModels.jl`, `Bellman.jl`, and
`DiscreteSolutions.jl`; exact filenames are implementation details.

The model supplies production/resources, utility parameters, the budget, and
economic constraints. The grid supplies numerical coordinates. The Bellman
implementation owns interpolation, global maximization, iteration, and residuals.
The solution owns immutable snapshots and policy evaluation.

Audit assumptions in `preferences`, `StateGrid`, default capital bounds, warm
starts, regridding, and plotting before reuse. Existing continuous-time
steady-state formulas must not silently determine discrete-time defaults. A
smooth deterministic interior steady state instead satisfies

```math
1 = \beta\left[f'(k^*) + 1-\delta_d\right].
```

For kinked technologies, branch-specific reference values are only proposals for
bounds; validate their branch membership and inspect boundary choices. Explicit
grid bounds remain available throughout development.

Do not put a discrete transition matrix in the existing `generator` field.
Generators have rows summing to zero; transition matrices have rows summing to
one. Shared operations should not erase this difference.

## Baseline global Bellman algorithm

Represent the value function by piecewise-linear interpolation in capital.
For fixed current capital, on each next-capital interval write

```math
\widehat V(k') = b_j + m_j k'.
```

The interval objective is

```math
u(B-k') + \beta(b_j + m_j k').
```

It is concave within the interval for positive CRRA risk aversion, even if the
objective across intervals is nonconcave. The maximization procedure is:

1. Intersect every interpolation interval with the state's feasible action set.
2. Evaluate feasible endpoints, respecting the utility domain.
3. For positive slope, evaluate the interior candidate if it lies in the interval:
   `c = (discount * m_j)^(-1 / γ)` and `k' = B-c`.
4. For nonpositive slope, use the leftmost feasible action in the interval.
5. Compare candidates globally and apply a deterministic exact-tie rule.

This finds the global optimum of the interpolated problem up to numerical error.
It does not establish that the grid approximation accurately resolves the
underlying continuous-state problem.

Start with value iteration. The reference improvement sweep costs approximately
`O(Nk^2)` in the deterministic case. Once it is verified, add modified policy
iteration to reduce the number of expensive improvement sweeps. Do not begin
with an Euler-equation solver, a local scalar optimizer over the entire action
range, or an endogenous-grid implementation that assumes global concavity.

For a fixed policy, interpolation defines a row-stochastic matrix `Pπ`, enabling
policy evaluation through `(I - discount * Pπ)V = uπ`. The matrix may remain
internal until public distribution evolution is implemented.

## Policy queries and kink handling

Default off-grid policy evaluation should recompute the current budget and solve
the one-period global maximization against the fitted continuation value.
Interpolating consumption across a policy jump can average two good actions into
a poor action and can violate feasibility when resources are nonlinear.

Insert known production crossings into grids when they lie within the chosen
bounds. Also inspect resource-rule kinks, such as income floors, when applicable.
Do not assume all value-function kinks coincide with production kinks: optimal
path switches and preimages of future kinks can create additional structure.

Initially refine manually around policy jumps and near-indifference. Report
competing distinct investment choices and their objective gap where practical;
adjacent numerical candidates representing the same optimum should not be
misreported as economically distinct branches. Do not impose a single-threshold
interpretation unless the computed dynamics support it.

Specify a deterministic exact-tie rule, provisionally choosing lower next-period
capital. Near ties should be diagnosed rather than silently treated as exact
ties. The rule can matter for paths at an indifference state.

## Boundaries and diagnostics

Distinguish the economic feasible set from the numerical domain. Restricting
choices to `[kmin, kmax]` imposes artificial bounds when those bounds are not
economic constraints. A positive numerical `kmin` must not be described as a
borrowing restriction.

Do not extrapolate continuation values silently. Report choices at numerical
bounds and require sensitivity checks that expand bounds and refine resolution
together. Log/CRRA singularities at zero require deliberate domain handling,
rather than interpolation through negative-infinite values.

Return at least convergence status, stopping reason, iteration count, update
norm, global Bellman residual, budget error, admissibility, and boundary choices.
Keep the last finite result on numerical failure where feasible, following the
package's existing ownership and failed-fit conventions.

For a finite, feasible approximation with finite rewards and nonnegative
interpolation weights, the Bellman operator is a contraction. Its residual gives

```math
\|V-V^*_{grid}\|_\infty
\leq \frac{\|TV-V\|_\infty}{1-\beta}.
```

This bounds iteration error for the approximate problem, not grid error or
threshold-location error. Recompute the residual using global maximization after
policy evaluation. Euler residuals are secondary diagnostics on smooth interior
branches and do not certify global optimality.

## Validation and milestones

### Milestone 1: deterministic correctness

1. Document constructor semantics, timing, budget, and utility domains.
2. Add discrete-time models, explicit grid support, and validation.
3. Implement global interval maximization and value iteration.
4. Add solution snapshots, diagnostics, warm starts, and off-grid policy queries.
5. Validate deterministic Ramsey and Skiba examples.

Use a smooth benchmark with an analytical policy where possible, such as log
utility with Cobb-Douglas production and full depreciation. Compare Bellman
updates against an independent dense action search, including solutions inside
intervals and at bounds. Check budget identities and feasibility directly.

For Skiba examples, vary resolution, bounds, and placement of the production
crossing. Study policy jumps, competing-path values, and deterministic capital
paths. Show which threshold estimates remain stable under refinement. Do not
infer scientific robustness from a small Bellman residual alone.

The milestone is complete when the examples are reproducible, numerical limits
are documented, and existing continuous-time tests still pass.

### Milestone 2: performance and integration

Add modified policy iteration and verify agreement with the reference solver.
Profile before adding search shortcuts. Integrate plotting and user-facing
examples. Avoid promising AD through policy switches.

### Future work, outside the first implementation

Add stochastic transition matrices, ability blocks, and distribution evolution
only after deterministic validation. Use the same interpolation weights for
backward valuation and forward probability movement.

Distribution work must address numerical dispersion across poverty-trap
boundaries, multiple closed classes, and periodic chains. An invariant
distribution need not be a pointwise limiting distribution in discrete time.
Evolution should advance by integer periods with an explicit observation contract.

Any future conversion from continuous time must explicitly define discounting,
depreciation, income accumulation, shock transitions, and decision timing. It
must not silently reinterpret an existing continuous-time model.

## Adopted implementation defaults

No unanswered question blocks the architecture or deterministic solver.
The implementation adopts these defaults:

- Nonnegative physical capital, with no borrowing or separate financial asset.
- Explicit log-utility support in the new implementation.
- Lower next-period capital as the exact-tie rule.
- Constructor names and parameter spelling as illustrated above.

## Numerical references

- [QuantEcon: Optimal Growth](https://julia.quantecon.org/dynamic_programming/optgrowth.html)
  describes fitted Bellman iteration for growth models.
- [QuantEcon: Discrete State Dynamic Programming](https://tools-techniques.quantecon.org/discrete_dp.html)
  covers value iteration, policy evaluation, and policy improvement.
- [Druedahl and Jørgensen: A General Endogenous Grid Method](https://researchprofiles.ku.dk/en/publications/a-general-endogenous-grid-method-for-multi-dimensional-models-wit-2/)
  provides context for later methods addressing nonconvexities and constraints.
