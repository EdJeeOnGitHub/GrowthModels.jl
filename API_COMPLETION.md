# API refactor acceptance audit

Validated locally on Julia 1.12.3. This audits `API_REFACTOR.md`, including the
requested downstream root migration; it does not certify full empirical
estimation jobs or remote CI platforms.

| Requirement | Implementation and evidence |
|---|---|
| Strict pre-refactor reference | `b0104e8` captures `067806e`, with provenance and a 300×20 stretched case. All 70 comparisons pass at `1e-11`, including exact same-environment iteration counts. |
| Compositional resources | Separate commit `cef5480`; custom resources and nested floors/transfers exercised in extension tests. |
| Model/process extension interfaces | `process` and `process_axis`; a model without conventional parameter/process fields and a categorical process defined only in tests complete solve, evaluation, regrid, stationarity, evolution, and AD. |
| Grid ownership and validation | Grids own coordinates, axis semantics, interpolation, regrid validation, weights, and preferred block size. Unknown option keys throw; custom OU bounds work without width. |
| Remove the numerical bridge | Workspace accepts `StateGrid`; one embedded-generator assembly path. Legacy state types, `flat_transfer`, duplicate diffusion assembly, and unused solver selection are removed. |
| Probability semantics | Stationary kernels consume/return mass. Population defaults, explicit initial distributions, transient absorption, ambiguous/zero-share classes, and share derivatives are tested. |
| Snapshot ownership | Read-only array wrappers; explicit editable copies; detached model inspection. 54 ownership assertions cover mutations/views, sparse operations, input isolation, serialization, warm starts, and scalar-read allocation. AD and plotting suites pass. |
| Cleanup and coverage | `test/legacy/` deleted after coverage migration; `test/COVERAGE.md` retains the audit. Unused Interpolations dependency/import removed. Historical reference generators remain intentional. |
| CI and documentation | CI includes Julia `1`, supported releases, and optional nightly; README execution added. README examples passed locally. README, migration notes, and validation records describe the resulting contracts. Remote CI was not run. |
| Root adoption | Root, filter, and joint-generative-model environments pin executable source revision `a6230e7`. Loaded source matches the checkout. 90 active callers/adapters are inventoried; the 390-assertion audit checks syntax, retired API use, literal includes, and environment pins. |
| Numerical changes separated | Selected-support bounds in `276354a`; bounds/resolution checks pass. Stable residual evaluation in `5126a3c`, verified against 256-bit arithmetic without relaxing tolerances. Both are documented separately. |
| Final validation | Full isolated package suite: 406 assertions. Eight downstream workflow suites: 142 assertions, plus the source audit. Recovery and filter suites also pass in their respective analysis environments. Before/after benchmarks and allocations are recorded in `API_VALIDATION.md`. |

The architecture criteria have direct evidence: extension types require no
solver/evolution edits; inspection cannot mutate owned storage through supported
properties; interpolation, embedded operators, block layout, and mass/density
conversion each have one owner. `getfield` is deliberately outside the supported
ownership contract.

## Downstream boundaries and result differences

The migration covers active API use, shared analysis adapters, and representative
workflows. Archived code under downstream `code/archive` and `sims/archive` is
preserved. The historical loader adapts an unchanged archived source and retains
its hash. Saved objects with retired types must be regenerated.

The downstream numerical recipes retain explicit warped grids and make analysis
interpolation/extrapolation explicit. Structural-panel sampling now uses cell
mass, retaining the old integer-horizon clock `0:T-1`. The analytical traps
example explicitly uses twelve unit steps from uniform cell mass. Forward and
backward threshold events now agree at exact threshold nodes. These analysis
changes are documented in the downstream migration notes.

Three exploratory notebook sections were already incomplete before migration:
`phase-plot.jl` references an undefined `eta_kdot_fn`; `growth-model-results.jl`
requires a manually selected `check_idx`; `kkt-counterfactuals.jl` has an unfinished
inverse-policy experiment (`evaluate_inverse`, `create_inverse_interpolations`,
and selection before initialization). Their migrated solver/inspection helpers
are tested, but the full exploratory cells are not claimed runnable. Implementing
those independent analysis experiments is not part of replacing GrowthModels
APIs. Full empirical estimates, saved-result regeneration, and publication of
the local Git pin were not performed.
