# Code review: scientific accuracy, performance, and software design

> Historical pre-redesign record. For the implemented API and current validation,
> see [README.md](README.md), [MIGRATION.md](MIGRATION.md), and
> [API_VALIDATION.md](API_VALIDATION.md).

Review date: 2026-11-03.

Several scientific correctness bugs should be fixed before relying on the package across its advertised models and grids. The sparse HJB approach is reasonable, but distribution evolution, nonconcavity handling, and several model helpers have substantive problems.

## Scope and validation

The review covered the source and tests and included isolated Julia reproductions. The full test suite could not run: loading the package failed because dependencies, starting with `Plots`, were missing. The isolated checks loaded selected source with minimal supporting type definitions; they were not end-to-end package tests. Performance opportunities below have not been benchmarked.

No implementation files were changed during the review. Line references describe the code at review time.

## Scientific correctness findings

### 1. [P1] Distribution evolution uses the wrong operator for densities

Location: [src/StateEvolution.jl](src/StateEvolution.jl), line 32.

The code applies `A'` directly to density `g`, then renormalizes. With cell weights `W`, the density operator should be `W⁻¹ A' W`; alternatively, evolve cell masses `p = Wg` using `A'`.

A two-state reproduction returned `[0.5, 0.25]` instead of `[0.6667, 0.1667]`. Both integrate to one, so normalization tests miss the error. More decisively, a density returned by `StationaryDistribution` changes after one evolution step.

This can distort wealth distributions and fixed-ability shares. The required transformation is also explained in [Achdou et al., numerical appendix §7.2](https://benjaminmoll.com/wp-content/uploads/2020/02/HACT_Numerical_Appendix.pdf).

### 2. [P1] Stochastic Skiba solvers do not resolve competing drift directions

Location: [src/HJB.jl](src/HJB.jl), lines 307 and 507.

With nonconcave value functions, `sf > 0` and `sb < 0` can both hold. The implementation then sets both indicators to one and `I0 = -1`. Consumption combines incompatible derivatives, while the generator includes both candidate drifts.

Select the admissible candidate with the higher Hamiltonian and construct consumption and the generator from that same policy. The deterministic implementation already attempts this; the stochastic versions omit it. See the corresponding selection in [Kaplan–Moll’s Skiba implementation](https://benjaminmoll.com/wp-content/uploads/2020/06/HJB_NGM_skiba.m).

### 3. [P1] The OU discretization can produce negative transition rates

Location: [src/HJB.jl](src/HJB.jl), line 190.

The drift always uses a forward difference. When drift is negative and diffusion is small, `zeta = mu/dz + s2/(2dz²)` becomes negative.

For `θ=0.1`, `σ=0.01`, and five grid points over `[0.8, 1.2]`, the stencil produced an off-diagonal rate of approximately `−0.09824`. Row sums can still be zero, so the existing check accepts an invalid generator.

Use sign-dependent upwinding and check off-diagonal nonnegativity as well as row sums.

### 4. [P1] Stationary distributions assume a unique communicating class

Location: [src/StateEvolution.jl](src/StateEvolution.jl), line 95.

Replacing one row only removes one null direction. Ability models deliberately contain disconnected blocks, each with its own stationary mass. A reproduction with two disconnected two-state blocks raised `SingularException`.

Solve stationary distributions within classes and preserve their initial weights, or require the caller to specify those weights. The current `StateEvolution` path requests this solve automatically for multidimensional models.

### 5. [P2] Skiba production derivatives select the wrong technology

Locations: [src/Skiba.jl](src/Skiba.jl), line 42; [src/StochasticSkiba.jl](src/StochasticSkiba.jl), line 45; the ability variant repeats the issue.

The derivative switches at `κ`, whereas production switches where the two technologies cross. With defaults, that crossing is approximately `2.69847`.

At `k=2.2`, the implemented derivative is `0.55533`; finite differencing production gives `0.06910`. This affects marginal-product and consumption-growth outputs. Stochastic derivative functions also use an array-valued condition in `if`, which fails for vector inputs.

### 6. [P2] Several reported economic quantities are mathematically incorrect

- [src/SolvedModel.jl](src/SolvedModel.jl), line 103: output growth uses `f(k) × k̇`; the chain rule requires `f′(k) × k̇`.
- [src/SmoothSkiba.jl](src/SmoothSkiba.jl), line 42: the high steady-state approximation adds `κ`, although this production function has no shifted capital term. At `β=0`, the actual unique steady state is approximately `1.78467`; the helpers report `1.29753` and `4.31566`.
- [src/HJB.jl](src/HJB.jl), line 452: the generic HJB diagnostic omits the stochastic generator contribution. For stochastic models, use the complete discrete residual `u(c) + A*v − ρ*v`.
- [src/HJB.jl](src/HJB.jl), line 403: `γ=1` divides by zero throughout initialization and utility evaluation. Implement log utility or explicitly reject that parameter.

## Other implementation issues

| Issue | Consequence | Location |
|---|---|---|
| Time-vector evolution uses absolute `t` rather than consecutive time differences | `[0,1,2]` advances by steps of 1 and 2 | [StateEvolution.jl](src/StateEvolution.jl), line 150 |
| Indexing accesses nonexistent `E_S` | `evolution[0]` throws `FieldError` | [StateEvolution.jl](src/StateEvolution.jl), line 18 |
| Poisson grid constructor references undefined `zmin` and `zmax` | Default construction fails for affected models | [utils.jl](src/utils.jl), line 177 |
| Supplied initial values are overwritten | Documented warm starts have no effect | [HJB.jl](src/HJB.jl), line 408 |
| OU sampling rounds the sample count independently of its time range | `T=1`, `dt=0.6` returns two times and three values | [StochasticProcesses.jl](src/StochasticProcesses.jl), line 82 |
| Exported two-asset model is incomplete | Its default-process path uses unsupported `λ`; the generic HJB expects absent fields/states | [StochasticTwoAssetKinked.jl](src/StochasticTwoAssetKinked.jl), line 15 |

## Performance opportunities

The strongest opportunities are structural. These are code-based opportunities, not measured speedup claims.

1. **Reuse factorizations during distribution evolution.** For fixed `A` and timestep, factor `I − Δt*A'` once and reuse it. Currently every step rebuilds and refactors the system.
2. **Reuse the primal factorization during differentiation.** [src/utils.jl](src/utils.jl), line 14, solves with the same `Areal` separately for the value and each derivative direction. Factor once and solve all right-hand sides through that factorization.
3. **Honor warm starts and remove the duplicated converged iteration.** Both solver loops recompute the same update from the previous value after detecting convergence.
4. **Preallocate HJB workspaces.** Grid spacings, repeated coordinate arrays, temporary policies, and sparse matrix structures are reconstructed repeatedly. Cache invariant data, use views, and update numerical entries in place.
5. **Exploit structure.** Deterministic systems are tridiagonal; fixed-ability blocks can be solved independently.
6. **Make performance-critical fields concrete.** `Function`, bare `NamedTuple`, and incompletely parameterized matrix/value fields inhibit specialization. Examples include [src/SolvedModel.jl](src/SolvedModel.jl), line 82, and [src/utils.jl](src/utils.jl), line 77.

## Assessment against A Philosophy of Software Design

The assessment is **partially aligned, but substantial redesign is warranted**. Model types and multiple dispatch provide a useful foundation. However, Ousterhout’s emphasis on simple interfaces and hidden implementation decisions is not consistently reflected here. [His design guidance](https://ramcloud.atlassian.net/wiki/spaces/RAM/pages/6848550) provides the relevant framework.

The clearest design problems are:

- Grid ordering, flattening, and density-versus-mass conventions leak across solver, evolution, and plotting code.
- Three large HJB implementations duplicate policy logic and have already diverged scientifically.
- `update_v` changes its return shape or meaning depending on convergence.
- Public types expose unsupported capabilities, while low-level solver functions unnecessarily expand the public interface.
- The README offers virtually no explanation of equations, assumptions, boundary conditions, or supported combinations.

A better organization would center on a grid object that owns indexing and cell weights, a shared policy-selection/generator kernel, and a consistent solver result. Distribution evolution should own the mass/density conversion. Plotting should sit outside the numerical core.

## Test quality and recommended next steps

The tests currently overstate confidence:

- [test/runtests.jl](test/runtests.jl), line 24, uses `!== zeros(...)`, which checks object identity rather than numerical contents.
- [test/test-precision.jl](test/test-precision.jl), line 48, permits policy variance up to `1e10`.
- [test/test-matlab-rbc-diffusion.jl](test/test-matlab-rbc-diffusion.jl), line 66, uses `abs(maximum(error))` instead of `maximum(abs.(error))`.
- Differentiation assertions are commented out; the two-asset test file is not included.
- Normalization-based mass tests cannot establish that the evolution equation is correct.

Fix the four P1 issues first, then add tests for stationary-density invariance, fixed-group mass preservation, generator positivity, policy/generator consistency, derivative accuracy, and grid refinement. Those checks should precede performance optimization.
