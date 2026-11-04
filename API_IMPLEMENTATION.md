# API implementation acceptance audit

Scope: API_BRAINSTORM.md and the GrowthCounterfactuals package in
`/home/ed/projects/aggregate-transfers/packages/GrowthCounterfactuals`.
Unrelated downstream changes and the root analysis environment are preserved.

| Requirement | Evidence | Status |
|---|---|---|
| Economics, callable production, preferences, population | Economics.jl; custom-model and downstream interpolant tests | Verified by final Pkg.test |
| Process owns axis generator, operator embeds layout | generator methods; custom process and old/new operator checks | Verified by final Pkg.test |
| Canonical StateGrid, axis metadata, weights, fixed/moving grids | Grids.jl; API and volatility AD tests | Verified by final Pkg.test |
| Stable finite solution including failed solves | Solutions.jl; genuine low-γ breakdown and policy replay | Ten failure assertions passed |
| Resolved settings and owned actual initial values | API exact replay, warm ownership, serialization | Verified |
| Explicit regrid, promotion, extrapolation rejection | API and regridded initialization AD tests | Verified |
| Strict unit mass, shaped mass/density copies | Probability and view-constructor tests | Verified |
| Required dt, private factors, observation invariance | Evolution.jl; shared-time and bracketing interpolation tests | Verified |
| Distinct trajectory/advance/observe, single trajectory storage | API tests; Evolution.jl storage audit | Verified |
| Stationary without dt, uniqueness/classes/transient absorption | API and generator tests, stationary AD | Verified |
| Fixed/moving grid, warm and stationary AD; no Base piracy | AD suite, internal nested-Dual solve; public surface audit | Verified |
| Hide retired APIs and unfinished two-asset model | GrowthModels.jl export audit; downstream absence assertions | Verified |
| Plotting extension | Five extension assertions; no Plots in numerical module | Verified |
| Breaking migration documentation and runnable examples | README, MIGRATION, check-readme.jl | README snippets passed |
| Numerical comparison to corrected pre-redesign implementation | Six explicit-grid cases vs f711991 fixture | 24 assertions passed |
| Retain scientific coverage with new interfaces | test/COVERAGE.md; kernel/scientific/refinement tests | Audited |
| Downstream overrides removed; economic floor and interpolant use standard solve | Two package source files; 20 integration assertions | Verified in durable environment |
| Downstream resolved dependency and actual dispatch | test/integration/Project.toml sibling sources; runtime path/method assertions | Verified |
| Separate numerical-default changes from interface changes | MIGRATION downstream grid section; capital-grid formula preserved | Documented |
| Repeated solves and streaming performance | benchmark-api.jl, API_VALIDATION.md | Verified; sequential timings in API_VALIDATION.md |
| Complete package-level tests and clean patch | Pkg.test and git diff --check | 237 assertions passed; clean patch |
| Final audit and reviewable commits | This acceptance audit; downstream commit 0d43c03d | Verified; core implementation committed with this audit |

## Environment and scope

The checked-in integration project adopts the local migrated GrowthCounterfactuals
package with this GrowthModels checkout and StochasticEM. Its generated Manifest
is ignored. The aggregate-transfers root Manifest remains pinned: historical
notebooks, separate modeling packages, and running analysis jobs are not migrated
as part of the named package integration. Use the new integration environment
for that package until those independent callers adopt the breaking API.

The original baseline checkout is `/tmp/growth-api-baseline`, commit f711991.
The comparison fixture and its regeneration script are retained in the repo.
Final timing claims use sequential runs after validation activity finished.

## First-target workflow evidence

1. Stochastic RCK solving, policy/boundary inspection, and transition/stationary
   moments: test-api and test-refinement (uniform and concentrated starts).
2. Real downstream interpolated production and income floor: integration test,
   with the standard solver and no overrides.
3. Finite preference-change warm start and volatility-driven coordinate rebuild
   plus explicit regrid: integration test, compared with fresh solves.
4. Stationary AD: downstream preferences, fixed/moving OU volatility, Poisson
   rates, and custom model preferences checked against finite differences.
5. Shared observations, conservation/refinement, finite failure policy and
   downstream rejection: API, refinement, kernel and failure tests.
6. Identical-grid old/new core comparison and repeated/moment-only performance:
   test-baseline and benchmark-api; numerical thresholds and timings in
   API_VALIDATION.md.
