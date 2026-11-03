# API redesign validation

## Correctness and integration

`GKSwstype=100 julia --project=. -e 'using Pkg; Pkg.test()'` passes all **237
assertions** on Julia 1.12.3. This uses Julia's isolated package test environment,
including the optional plotting extension. CI now starts at Julia 1.9, the
minimum version supporting the extension configuration; other CI versions were
not run locally.

The suite checks custom economic/preferences/process extensions, strict unit
mass, fixed observation schedules and interpolation refinement, generator and
class-absorption invariants, boundary constraints, finite failed-solve snapshots,
initial-value ownership, serialization, fixed/moving-grid AD, regridded warm
starts, sparse Hessians, and policy/transition/stationary derivatives against
finite differences. The coverage migration is mapped in
`test/legacy/COVERAGE.md`.

Six explicit-grid cases compare the new implementation against commit f711991:
RCK, Skiba, SmoothSkiba, OU RCK, Poisson Skiba, and ability Skiba. Value and
consumption differences satisfy `1e-7`; forward probability differences satisfy
`1e-8`. These are test thresholds, not claimed maximum observed differences.
Default-grid changes are deliberately excluded from this comparison.

The real GrowthCounterfactuals package passes **20 integration assertions** using
its callable production interpolant and income floor, standard solver dispatch,
fixed population shares, warm starts, value interpolation, and a stationary
moment derivative checked against finite differences. The checked-in
`test/integration/Project.toml` resolves sibling working checkouts and verifies
the actual loaded source. See `test/integration/README.md` for commands.
The downstream root analysis environment remains separate and pinned.

`julia --project=. test/check-readme.jl` executes all README Julia snippets
successfully. `git diff --check` passes.

## Performance

`test/benchmark-api.jl` runs on either this checkout or f711991 using identical
explicit grids and initial probabilities. Measurements below are medians of
three runs after compilation, collected sequentially after the validation runs.
Solve timings include setup. New trajectory timing reuses a constructed
Evolution; the old high-level call constructs its workspace each time. Thus
trajectory timings compare these public workflows, including their setup
contracts, rather than identical low-level factorization work. A separate
unrelated analysis process was running on the machine; small timing differences
should not be treated as guarantees.

| Case | Before | New | Before allocation | New allocation |
|---|---:|---:|---:|---:|
| Skiba solve, 1000 points | 7.32 ms | 7.94 ms | 0.57 MB | 1.18 MB |
| Skiba trajectory | 1.35 ms | 1.23 ms | 1.64 MB | 0.21 MB |
| OU solve, 200 × 20 | 84.82 ms | 78.88 ms | 50.98 MB | 48.17 MB |
| OU trajectory | 30.26 ms | 22.49 ms | 10.65 MB | 0.81 MB |

Iterations agree: 56 for Skiba and 12 for OU. The new contract owns more
reproducibility data; deterministic solve allocations increase and this case is
about 8% slower. OU solve time is about 7% lower and its reusable trajectory
workflow about 1.35× faster. There is no across-the-board speedup over the already
optimized pre-redesign implementation.

Additional new-API OU workflows:

| Operation | Median time | Allocation |
|---|---:|---:|
| Same-model warm solve | 12.42 ms | 10.24 MB |
| Stream first capital moment | 23.04 ms | 0.80 MB |
| Final distribution only | 21.37 ms | 0.23 MB |

Streaming avoids retaining trajectory history; these figures report total
allocations, not peak retained memory. All evolution cases use `dt=0.1` over
20 units of time; trajectory/moment output is requested at integer times.
Warm-start speed depends on proximity of the models, not merely grid reuse.

## Result differences and limits

The core comparisons hold grids fixed. Arbitrary observation times now use the
fixed-lattice interpolation contract and can differ from the previous
endpoint-dependent stepping. GrowthCounterfactuals now uses analytical central
98% OU bounds instead of simulated quantiles; this separate default change can
change its numerical results and is documented in MIGRATION.md. Default-grid
analyses must be regenerated. The capital split formula is preserved.

Derivatives at economic kinks or changes in stationary communicating classes
are not guaranteed smooth. Log utility and the unfinished two-asset model remain
unsupported. Incremental integration and estimation frameworks are deferred by
the accepted design.
