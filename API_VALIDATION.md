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
`test/COVERAGE.md`.

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

## Deep-module refactor validation (2026-11-03)

The refactor's full isolated `Pkg.test()` run passed **406 assertions** with
`GROWTHMODELS_STRICT_ITERATIONS=true` on Julia 1.12.3. This includes all 70
same-environment comparisons against `067806e`, population default/override
checks, read-only storage, migrated historical invariants, default-bound checks,
and a 256-bit residual cross-check. README examples passed. GrowthCounterfactuals
integration passed 20 assertions. The broader root analysis migration has separate workflow and source-audit
evidence below; package tests alone do not certify analysis callers.

The first full run caught an import ambiguity between Plots and GrowthModels
for `density` in a newly added test. Plotting tests now import Plots without
polluting the test namespace, and probability tests qualify the evaluator.

Before/after workflow timings were collected with the existing benchmark script.
Other validation/precompilation jobs ran during this work, so these timings are
indicative, not isolated performance guarantees:

| Workflow | Before | After | Before allocation | After allocation |
|---|---:|---:|---:|---:|
| Skiba solve | 12.34 ms | 8.64 ms | 1,176,616 B | 1,145,608 B |
| Skiba trajectory | 1.95 ms | 1.33 ms | 210,720 B | 210,368 B |
| OU solve | 141.10 ms | 106.80 ms | 48,161,808 B | 48,123,336 B |
| OU trajectory | 31.74 ms | 29.58 ms | 805,208 B | 804,744 B |
| OU warm solve | 21.92 ms | 12.96 ms | 10,236,264 B | 10,201,120 B |

Iteration counts remained 56 (Skiba) and 12 (OU). A loop of 1000 public scalar
reads from `fit.value` took about 5.9 microseconds and allocated 16 bytes, avoiding
whole-array copies. Sparse multiplication and reductions delegate to backing
storage while public mutation remains forbidden.

The residual investigation distinguished two cases. On a 900-node RCK grid
with bounds `(0.001,20)` and power 3.9, the former direct computation reported
`1.31e-6` while 256-bit arithmetic gave `5.53e-7`: a false iteration-limit failure.
Value-difference evaluation fixes it without changing tolerances. On the more
extreme 1000-node/power-4 grid, high precision also found residual above tolerance;
that numerical difficulty must not be hidden by simply weakening convergence.

Root-environment integration now passes 8 mainline structural assertions, 13
analytical-export/probability assertions, and 8 metastability/PCCA assertions.
The installed executable GrowthModels source matches this checkout. These
small synthetic cases validate the migrated callers, not full empirical
estimation or figure jobs. The separate joint-generative-model environment is
also pinned. Historical-loader integration passes 18 assertions against an
unchanged archive and current source; the migrated continuous-shock recovery
example passes 8 assertions in the root environment. Recovery and filter
integration also pass in their separate environments.
Notebook solver/prediction/profile/sampling checks pass 43 assertions; filter
simulation and analysis checks pass 24. The 90-file active caller inventory
passes 390 syntax, retired-API, literal-include, and dependency-pin assertions.
See `API_COMPLETION.md` for the distinction between migrated API workflows and
pre-existing incomplete exploratory cells. The eight downstream workflow suites
contain 142 assertions in total, excluding repeated environment runs and the
source audit.
