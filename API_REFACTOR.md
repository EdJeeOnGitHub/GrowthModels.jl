# API refactor implementation plan

Status: implemented and locally validated, including active downstream API
migration. See [API_COMPLETION.md](API_COMPLETION.md) for the requirement audit,
verification scope, and pre-existing exploratory notebook limitations.

Implement in this order: capture baselines → resource composition → model/process
interfaces → grid semantics → numerical bridge removal → probability semantics
→ ownership protection → final cleanup and integration. Each step should leave
the package usable and independently testable.

“Preserve the workflow” means the model → grid → solution → evolution flow, not
compatibility with retired APIs. Numerical-default and convergence-policy changes
must remain separate from structural refactoring.

## Design objective

Each module should own a complete set of decisions and hide its representation.
Economic rules, process semantics, grid layout, solver machinery, and probability
evolution should each have one owner.

```text
Economics ──┐
            ├─ Grid resolution ── HJB solve ── Solution snapshot
Processes ──┘                                     │
                                       Probability dynamics
```

Retain the public model → grid → solution → evolution workflow. Keep the scope
to one capital state, one optional productivity process, and optional fixed
ability types. These are conceptual modules; each does not need a Julia
submodule.

Source links below identify existing files; function names identify the change
sites even as line numbers move.

## 0. Capture a strict structural-refactor baseline

Before changing behavior, capture a reproducible baseline against commit
`067806e`. Retain the existing pre-redesign fixture and its scientific comparison
in [test-baseline.jl](test/test-baseline.jl); add a separate refactor baseline,
generation script, and provenance under `test/test-data/`.

- Use explicit coordinates and identical settings in both implementations.
- Record values, consumption, drift, resources, generators, evolved
  probabilities, convergence status, and iteration counts.
- Add a larger stretched-capital-grid case alongside the small existing cases.
- Target approximately `1e-11` absolute agreement on controlled reference cases;
  record justified scale-aware tolerances for cases whose magnitudes require
  them. Do not silently relax thresholds to make a refactor pass.
- Require matching iteration counts in same-environment before/after checks.
  Across Julia versions and linear-algebra implementations, investigate
  differences rather than treating exact iteration equality as universal.
- Record Julia version, numerical settings, and environment with the baseline.

The reported worst current fixture discrepancy of about `1.7e-13` is feedback
from an external review, not independently verified evidence. Reproduce the
measurement before using it to calibrate the gate. Existing scientific and AD
checks remain necessary in addition to close numerical agreement.

## 1. Change resource composition in a separate commit

Change `resources(::ResourceModel, state)` in [Economics.jl](src/Economics.jl) to:

```julia
resources(m::ResourceModel, state) =
    max(resources(m.model, state) + m.transfer, m.income_floor)
```

This makes nested adjustments compose and preserves custom base resource rules.
The current documented formula starts from production, so this is an intentional
economic-contract change, not merely an implementation fix. Preserve ordinary
built-in behavior and document changed custom/nested-wrapper behavior in
[MIGRATION.md](MIGRATION.md).

**Acceptance:** Test a custom resource adjustment, nested transfers and floors,
and unchanged ordinary built-in resources. Keep this behavior-changing commit
separate from structural comparisons; identify affected cases explicitly.

## 2. Establish a complete model and process interface

Change [Economics.jl](src/Economics.jl),
[StochasticProcesses.jl](src/StochasticProcesses.jl), and the exports in
[GrowthModels.jl](src/GrowthModels.jl).

- Introduce `process(model)`, returning `nothing` for deterministic models.
- Implement explicit delegation for `ResourceModel`: preferences, production,
  process, population, and grid suggestions.
- Remove `ResourceModel`'s arbitrary `Base.getproperty` forwarding.
- Derive supported state schemas from process presence and population support.
  Keep the existing three supported schemas.
- Add a process-axis extension method that resolves numerical options into
  coordinates plus one semantic distinction: interpolable or categorical.
  The process implementation enforces its own support restrictions; do not add
  a separate fixed/adjustable trait without a concrete use case.
- Keep `generator(process, coordinates)` responsible only for the axis
  generator.

The process-axis method owns process-specific decisions. Poisson returns its
fixed support; OU resolves coordinates or bounds and validates its
discretization. Custom processes implement the same contract.

**Acceptance:** A custom model with no `γ`, `ρ`, `δ`, or `stochasticprocess`
fields works through its interface. Wrapping it preserves its economic rule.

Extend [test-extensions.jl](test/test-extensions.jl) with these cases and a custom
categorical process. Define both the model and process only in the test file,
and exercise solve, policy evaluation, regridding restrictions, evolution, and
stationarity through public methods. No production-code special cases may be
added for these test types.

## 3. Make the grid the sole owner of layout and interpolation semantics

Refactor [Grids.jl](src/Grids.jl), including `StateGrid`, `compatible_grid`, and
`validate_model_grid`.

Introduce one internal resolution path with explicit precedence:

1. Explicit coordinates.
2. Explicit bounds and resolution.
3. Model/process defaults.

Evaluate defaults only when needed. This fixes the eager OU fallback in
`get(zspec, :bounds, productivity_bounds(p, zspec.width))`, which currently reads
`width` even when bounds are supplied. This failure affects custom defaults
without `width`; built-in defaults currently merge it in.

Validate supplied option keys before merging defaults. Reject unknown keys such
as `k=(n=50, bound=(0.1, 5.0))` instead of silently ignoring them. Validate
process-specific options within the owning process resolver, with errors naming
the unsupported key and supported alternatives.

Store resolved axis semantics alongside coordinates. Separate “Markov state in
probability calculations” from “categorical state that cannot be interpolated.”
OU and Poisson currently share `:markov`, which is insufficient.

Move `interpolate_grid` from [Solutions.jl](src/Solutions.jl) into the grid
implementation. Add shared internal operations for:

- Validating evaluation states.
- Checking exact grid compatibility.
- Checking whether regridding is permitted.
- Interpolating continuous axes while selecting exact categorical states.

Simplify `validate_evaluation` and `regrid` to use those operations. Remove their
`isa PoissonProcess` checks.

Preserve current default capital-bound behavior in this structural step.
Resolving default capital bounds against selected productivity/ability support
belongs to a separate numerical-default change, described below. Explicit
capital bounds remain authoritative.

**Acceptance:** Custom categorical processes receive the same support
protection as Poisson; custom OU bounds need no `width`; keyword order does not
affect layout; unknown option keys throw; fixed/moving-grid AD remains correct.

Add these cases to [test-api.jl](test/test-api.jl),
[test-extensions.jl](test/test-extensions.jl), and
[test-api-ad.jl](test/test-api-ad.jl).

## 4. Remove the legacy numerical representation

Replace `numerical_state` in [Grids.jl](src/Grids.jl). It currently translates:

```text
StateGrid → StateSpace → StateSpaceHyperParams → HJBWorkspace
```

Use `StateGrid` directly as the geometry source. Build resource arrays and the
embedded process generator inside the HJB setup layer.

Change:

- [HJBWorkspace.jl](src/HJBWorkspace.jl): construct `HJBWorkspace` from the model,
  grid, and required numerical scalar type.
- [Solutions.jl](src/Solutions.jl): simplify `solve` setup.
- [HJB.jl](src/HJB.jl): initialize values from resolved resources and preferences
  in `initial_guess`.
- [HJB.jl](src/HJB.jl): retire duplicate `construct_diffusion_matrix`
  implementations after migrating their tests.

Keep one internal operator assembly function responsible for embedding axis
generators and independent ability blocks. Keep mutable buffers in
`HJBWorkspace`. Centralize the layout-derived preferred block size currently
computed as `prod(dims[1:2])` in both `HJBWorkspace` and `Evolution`; both should
consume one internal layout helper. Retain the actual coupling validation in
`independent_blocksize` rather than assuming every proposed block is independent.

Remove the workspace's legacy `flat_transfer` argument and alternative
diffusion-construction path. Economics already supplies resources.

**Acceptance:** Meet the strict same-environment baseline gate from step 0,
including iteration counts, while preserving failure snapshots, scientific
invariants, and AD behavior.

Update the operator tests in [test-kernels.jl](test/test-kernels.jl). Replace
old/new assembly comparisons in [test-extensions.jl](test/test-extensions.jl)
with independent expected matrices and generator invariants before deleting the
old implementation.

## 5. Consolidate probability evolution around mass

Keep the public `trajectory`, `advance`, `observe`, and `stationary` functions.

Refactor [StateEvolution.jl](src/StateEvolution.jl) so the stationary kernel
accepts a generator and optional initial **mass**, and returns mass. Currently
`stationary` in [Evolution.jl](src/Evolution.jl) adapts to the old density API by
supplying all-one weights.

- Keep density conversion at the distribution boundary.
- Centralize forward-generator orientation and validation.
- Preserve the shared observation walker and fixed integration lattice.
- Preserve initial class shares and transient absorption.
- Remove unused weighted-normalization helpers after checking callers.

Make `Population.shares` an actual default for `stationary(fit)`:

- `stationary(fit, initial)` always honors the explicit joint distribution,
  including its ability-class masses and transient absorption.
- Without an initial distribution, use population shares when each positive-share
  ability block has one closed communicating class and thus a unique limiting
  distribution within that block.
- If a positive-share block has multiple closed classes, throw and require an
  explicit initial distribution; do not invent within-type mixture weights.
- Zero-share blocks contribute zero mass and need no arbitrary stationary mixture.
- Without ability types, retain the unique-closed-class requirement.

An explicit uniform start producing equal ability shares is correct even when
population defaults differ. Never overwrite explicit probabilities with defaults.
Implement class analysis and weighting in the probability layer, not the solver.
No conditional-distribution convenience constructor is needed in this refactor.
Document this new default separately from the mass-kernel structural change.

**Acceptance:** Probability conservation, stationary mixtures, observation
invariance, temporal refinement, and initial-probability derivatives continue to
pass. Add cases for default `[0.3, 0.7]` shares, explicit overriding shares,
multiple classes within a type, zero-share ambiguous blocks, transient states,
and differentiation with respect to shares in a fixed class topology.

Use [test-kernels.jl](test/test-kernels.jl),
[test-refinement.jl](test/test-refinement.jl), and the AD suite.

## 6. Enforce snapshot ownership with read-only array views

Standardize ownership across:

- `GrowthSolution` and `ValueGuess` in [Solutions.jl](src/Solutions.jl).
- `StateDistribution` in [Distributions.jl](src/Distributions.jl).
- `Evolution`, `Trajectory`, and `Observations` in
  [Evolution.jl](src/Evolution.jl).
- `Population` in [Economics.jl](src/Economics.jl).

Retain useful inspection syntax such as `fit.value`, `fit.policy`, and
`fit.diagnostics`, but expose array storage through thin read-only wrappers.
Property reads must not copy whole arrays, and attempted mutation must throw
rather than modify a discarded copy:

```julia
fit.value[i]          # Read without copying the backing array
fit.value[i] = 0      # Throws
editable = copy(fit.value)
```

Keep existing explicit `mass` and `density` copy-returning contracts. Protect
arrays in policy/diagnostic containers and trajectory times as well as values.
A read-only wrapper must not expose a supported mutable backing-array accessor.
Preserve input ownership so later mutation of constructor inputs cannot change
stored results.

Keep an owned internal model snapshot. Public model inspection may return a
detached copy, rather than introducing a general recursive read-only object
system. Internal policy evaluation must use the owned model so its resources
remain consistent with the stored policy and generator. Audit mutable processes
and callable production objects at this boundary.

Keep `Evolution` factors and workspace storage outside the supported public
property interface. Define explicit `propertynames` contracts and prevent
ordinary positional constructors from bypassing validation or ownership.

Retain the internal-access audit: wrappers can affect dispatch into sparse,
structured, interpolation, and AD code. Internal kernels may use private storage
helpers or `getfield` when required. Verify performance rather than promising
zero overhead. Julia's deliberate `getfield` escape hatch is outside the
supported interface.

Check indexing, views, broadcasting, mutation through views, sparse operations,
`copy`, warm starts, serialization, and AD. Update
[GrowthModelsPlotsExt.jl](ext/GrowthModelsPlotsExt.jl) as needed to use the final
inspection contract.

**Acceptance:** Supported writes through read-only arrays fail clearly; explicit
copies remain editable; inspecting or modifying detached model data cannot
change policy evaluation or evolution. Repeated scalar indexing has no
whole-array allocations. Benchmark both external property-indexing loops and
internal numerical workflows in [benchmark-api.jl](test/benchmark-api.jl).

Do this after the structural changes so protection covers the final types. It
may be delivered separately, but the snapshot-integrity acceptance criteria
remain incomplete until it lands.

## 7. Delete obsolete machinery and align documentation

Once the new paths pass:

- Remove `StateSpace`, `HyperParams`, `StateSpaceHyperParams`, and their
  constructors from [utils.jl](src/utils.jl).
- Remove the remaining legacy constructors in
  [SmoothSkiba.jl](src/SmoothSkiba.jl).
- Remove unused `hjb_linear_solver` in
  [HJBWorkspace.jl](src/HJBWorkspace.jl).
- Move differentiable factorization helpers from `utils.jl` beside
  [LinearSolvers.jl](src/LinearSolvers.jl).
- Move quadrature helpers beside grids and process-bound helpers beside
  processes.
- Remove the unused `Interpolations` import from
  [GrowthModels.jl](src/GrowthModels.jl) and dependency from
  [Project.toml](Project.toml), after checking package and extension consumers.
  Downstream packages using it must declare their own dependency.
- Update include order and audit exports in
  [GrowthModels.jl](src/GrowthModels.jl).

Preserve historical baseline-generation scripts that intentionally run against
an older checkout; their legacy references are not current-package dependencies.
In particular, review the historical branches in
[benchmark-api.jl](test/benchmark-api.jl) and
[generate-baseline.jl](test/integration/generate-baseline.jl) before removing
references.

Delete `test/legacy/` after verifying migrated coverage. First audit the
[coverage mapping](test/COVERAGE.md) (originally `test/legacy/COVERAGE.md`), port any missing tests for supported
behavior, and record intentionally unsupported historical behavior. Preserve the
completed coverage mapping outside the deleted directory, and update references
to the archive. Deletion follows successful validation of the migrated tests;
old API usage alone is not evidence that coverage is redundant.

Update [README.md](README.md), [MIGRATION.md](MIGRATION.md), and the validation
documents with the actual final contracts. Document the custom process
interface, ownership rules, resource composition change, and population-share
semantics.

## 8. CI and downstream integration

Update [.github/workflows/CI.yml](.github/workflows/CI.yml) to include the current
stable Julia selector in addition to supported-version coverage, and execute
[test/check-readme.jl](test/check-readme.jl) in a suitable instantiated environment.
Keep nightly failures distinguishable from supported-release regressions.

Audit the downstream GrowthCounterfactuals package before removing forwarding.
In `/home/ed/projects/aggregate-transfers/packages/GrowthCounterfactuals/src/ModelSolver.jl`,
`create_model_grid` currently reads `m.stochasticprocess` (line 147 at review).
Migrate this to `process(m)` and search for other assumptions about wrapper fields.
Run [growthcounterfactuals.jl](test/integration/growthcounterfactuals.jl) against the
actual migrated source, with runtime checks of the resolved package location.

Include migration of the broader `/home/ed/projects/aggregate-transfers` root
analysis environment and its callers in this implementation scope. Inventory
scripts, notebooks, and packages using GrowthModels; migrate supported callers
and validate representative analysis workflows in the root environment as well
as the isolated integration environment. Preserve unrelated working changes.

The downstream root Project uses `rev = "main"` for GrowthModels. Its Manifest
records a concrete tree, but dependency updates can advance that source. Update
the Project and Manifest deliberately to the migrated API, pin the intended
GrowthModels revision, and verify the actual loaded source. Keep unrelated
dependency upgrades out of this migration. Record affected entry points,
validation results, and any expected result differences in the migration notes.
Include an old-to-new API mapping in [MIGRATION.md](MIGRATION.md), covering model
process access, explicit grid construction, solution inspection, and probability
evolution. Identify saved results that must be regenerated because their retired
types cannot be loaded under the new API. Keep archival source immutable and
migrate active loaders or callers that still depend on it.
Downstream migration is not complete with only the isolated package test: the
root environment and migrated callers must also be validated.

## Separate numerical work

These changes must not be hidden in structural-refactor commits:

1. **Capital-bound defaults:** Resolve bounds against selected productivity and
   ability support only in a separate change with a migration note and explicit
   before/after default-grid tests. The fixed-coordinate baseline cannot detect
   this change. Include bounds/resolution refinement and downstream impact.
2. **Stretched-grid residual failures:** First reproduce the reported false
   `:iteration_limit` result in `solve`/`nonlinear_residual` in
   [Solutions.jl](src/Solutions.jl). Distinguish discretization error, conditioning,
   cancellation, and genuine nonconvergence. Validate against an independent
   reference or precision study before changing residual evaluation or stopping
   criteria. Do not loosen tolerances just to make the refactor pass.

## Final validation and completion criteria

Run the full package suite, README examples, downstream integration, and
before/after benchmarks. Relevant entry points:

- [runtests.jl](test/runtests.jl).
- [check-readme.jl](test/check-readme.jl).
- [test-baseline.jl](test/test-baseline.jl).
- [test-failure.jl](test/test-failure.jl).
- [Integration instructions](test/integration/README.md) and
  [growthcounterfactuals.jl](test/integration/growthcounterfactuals.jl).
- [benchmark-api.jl](test/benchmark-api.jl).

Apply the strict step-0 refactor gate and preserve scientific invariants. Report
allocation or timing regressions introduced by ownership changes rather than
assuming that architectural cleanup improves performance.

Architecture acceptance is concrete:

- A new supported process requires no edits to solution evaluation or evolution.
- A new economic model requires no edits to the solver.
- Public inspection cannot silently invalidate an existing solution or
  distribution.
- Grid semantics, operator assembly, and probability conversion each have one
  implementation owner.

## Confirmed user decisions

1. **Historical tests:** Delete `test/legacy/` after migrating and verifying
   supported coverage; preserve the coverage audit outside that directory.
2. **Downstream root environment:** Include migration of the root analysis
   environment and its scripts, notebooks, and package callers, in addition to
   GrowthCounterfactuals integration. Include migration documentation and
   representative workflow validation; an API replacement list alone is not
   sufficient.

The plan adopts compositional resource rules, population-weighted stationary
defaults, and read-only array views as described above; these are no longer
open implementation choices.

## Implementation evidence (2026-11-03)

- Baseline commit `b0104e8` captures `067806e` on Julia 1.12.3, including the
  300 × 20 stretched case and same-environment iteration counts.
- Resource composition is isolated in `cef5480`.
- Structural contracts, read-only snapshots, probability defaults, dependency
  cleanup, and migrated legacy coverage are in `43ef54d`.
- Selected-support capital bounds are isolated in `276354a`.
- High-precision-verified residual evaluation is isolated in `5126a3c`.
- Full `Pkg.test()` passed all 406 assertions with
  `GROWTHMODELS_STRICT_ITERATIONS=true`; README
  examples passed. The first full run exposed a Plots/GrowthModels `density`
  import ambiguity in the new tests, which was corrected before the passing run.
- GrowthCounterfactuals integration passed 20 assertions against this checkout.
- Scalar inspection benchmark: 1000 reads of `fit.value[i]` allocated 16 bytes,
  without whole-array copies. Numerical benchmark iteration counts remained 56
  (Skiba) and 12 (OU); timings are workload-dependent.
- `test/legacy/` was deleted after coverage migration; `test/COVERAGE.md` retains
  the audit and references to the historical revision.
- Root migration is implemented. Its Project/Manifest now use a pinned local Git
  revision of GrowthModels instead of `main`. Resolving required replacing
  unavailable `Qt6Base_jll 6.10.2+1` with `6.10.2+2`; adding StochasticEM also
  required AdvancedHMC 0.8.3 → 0.8.5. JointModel loading required aligning
  PCNMCMC with the analysis environment's existing `50ead01` pin.
  Removed obsolete GrowthModels dependencies may leave the resolved graph.
- Mainline structural solvers and counterfactual/fitting scripts have been
  migrated; mainline integration passed 8 assertions in the root environment.
- The analytical Skiba export and traps notebook share a migrated probability
  module. Its 13 integration assertions cover exports, mass conservation,
  forward/backward duality, and exact-node threshold semantics.
- The joint-generative-model environment is also pinned to `a6230e7`.
- The PCCA notebook retains its warped grid and passes 8 integration assertions,
  including clustering. Optim and Arpack are now explicit root dependencies
  at their existing resolved versions. Installed GrowthModels executable source
  was checked against this checkout.
  The remaining notebook API calls have been migrated. A 90-file active caller
  inventory passes 390 source/environment assertions; notebook workflows pass
  43 assertions and filter simulation/profile checks pass 24.
- Historical-loader integration passes 18 assertions for unchanged archived
  source and the current recovery source. The continuous-shock recovery example
  passes 8 assertions in the root environment, preserving historical linear
  extrapolation as an explicit analysis approximation. The analysis declares
  Interpolations directly; it is not restored as a core dependency.

- Recovery and filter integration also pass in their separate analysis environments.
  The filter environment now pins the same GrowthModels revision. No full
  empirical estimation or exploratory figure job is claimed by these checks.
