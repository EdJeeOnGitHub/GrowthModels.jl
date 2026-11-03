# Performance and interface changes

These are historical measurements of the performance implementation committed as
`614172c`. Subsequent correctness changes alter numerical defaults; see
[CORRECTNESS_REVIEW.md](CORRECTNESS_REVIEW.md) before comparing new results.

Validated on 2026-11-03 against commit `caec88c`, which already contains the
numerical correctness fixes. These comparisons isolate the performance changes
from those earlier scientific corrections.

## Implemented changes

1. Forward evolution caches implicit-Euler factors and solve buffers. Generator
   or timestep changes refresh numerical factors; sparsity changes rebuild them.
2. HJB solves honor copied warm starts and return the converged update without
   repeating the final solve.
3. Forward output can retain requested observations, only the final density, or
   stream moments through a callback without keeping a trajectory.
4. Stationary distributions are computed separately or explicitly requested.
5. HJB geometry, policy buffers, and sparse storage are allocated once per solve.
6. Differentiation shares primal factors across tangent directions, including
   nested dual numbers.
7. Deterministic HJB systems use tridiagonal elimination; fixed-ability models
   use independent sparse blocks. Stable sparse patterns permit symbolic reuse.

The HJB implementations now share one iteration loop and policy-selection
routine. `HJBWorkspace.jl`, `EvolutionWorkspace.jl`, and `LinearSolvers.jl`
separate numerical storage, evolution, and linear algebra. Public workspace
methods own input copies and define cache invalidation; callers do not need to
manage factorization details. See [README.md](README.md) for usage examples.

## Measurements

Julia 1.12.3; median of three runs after compilation. HJB grids have 200 capital
points, 21 productivity points where applicable, and three ability groups.
Forward paths contain 100 steps of length 0.25. Timings include setup and
factorization. Baseline and optimized benchmarks ran sequentially, although
unrelated Julia workloads were active on this machine; timings are indicative,
not performance guarantees. Allocations below are cumulative allocated bytes,
not peak resident memory.

| HJB case | Before cold | After cold | Cold speedup | Before warm | After warm |
|---|---:|---:|---:|---:|---:|
| Deterministic Skiba | 1.353 ms | 0.694 ms | 1.95× | 1.350 ms | 0.257 ms |
| OU RCK | 74.984 ms | 54.641 ms | 1.37× | 73.278 ms | 15.991 ms |
| Fixed-ability Skiba | 469.355 ms | 295.189 ms | 1.59× | 458.590 ms | 45.715 ms |

Warm timings reuse the same model's converged value. The old solver ignored the
guess; the new solver converges in one update. Parameter changes can require
more iterations. Cold allocations fell by 96.5%, 47.8%, and 51.3%, respectively.

| Density case | Uncached | Cached, full output | Speedup | Allocations, uncached → cached → final-only |
|---|---:|---:|---:|---:|
| OU RCK | 800.0 ms | 21.2 ms | 37.7× | 592.0 → 15.4 → 9.0 MiB |
| Fixed-ability Skiba | 2282.3 ms | 63.1 ms | 36.2× | 1641.7 → 46.3 → 27.1 MiB |

Final-only timings were 18.9 ms and 59.2 ms. The uncached reference repeatedly
calls the weighted-density step with the same generator; stationary computation
is excluded from both paths.

A sparse derivative problem with 300 states and six tangent directions improved
from 2.178 ms to 0.379 ms (5.74×), with allocations falling from 3.576 MiB to
0.631 MiB.

## Numerical agreement and tests

The complete suite passes **431,965 assertions**, including **110 new assertions**
for caching, interfaces, warm starts, structured solves, and differentiation.

Before/after cold HJB value functions and consumption policies differ by at most
1.1e-12 across the three benchmark cases. Cached density paths differ from their
uncached references by less than 1e-12. Sparse Jacobians differ by at most
1.2e-16. Separate tests compare Jacobians and Hessians with finite differences.
These are checks on the tested cases, not a proof of identical behavior for
every parameterization. Warm and cold solves can stop at different iterates;
tightening `crit` improves their agreement.

## Changes callers should know

- `stationary_dist` is now `nothing` by default. Request `stationary=true` or
  call `StationaryDistribution(path)` separately.
- Explicit zero initial guesses are honored. Use `init_value=nothing` or omit
  the keyword for the default guess. Warm-start inputs are never mutated.
- Real observation times label the initial state at the first time and advance
  by successive intervals. Integer `T` retains observations at `0:T-1`.
- Callbacks receive borrowed cell-probability buffers. Copy before retaining
  them, and use a separate workspace for concurrent simulations.
- Dual-number convergence checks include tangents. Grid construction avoids
  spurious NaN derivatives at zero endpoints and correctly normalizes grid
  endpoints when a nonzero warp coefficient has zero power. That last edge
  case intentionally changes previously incorrect grid locations.

Reproduce tests and benchmarks using the commands in [README.md](README.md).
