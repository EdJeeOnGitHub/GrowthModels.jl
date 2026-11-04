# Coverage migration audit

| Historical coverage | Current evidence / disposition |
|---|---|
| Weighted mass and nonuniform geometry | Probability tests in test-api; nonuniform drift and conservation in test-kernels; mixed-axis distributions and AD |
| Arbitrary nonunit measures | Deliberately removed: constructors reject nonunit probability input |
| Stationary classes and transient absorption | test-api ability shares; test-kernels absorption; initial-probability AD |
| Exclusive Hamiltonian selection | test-kernels explicit forward/backward/zero choices and generator drift |
| OU positivity and Ito drift | test-kernels zero/small/ordinary volatility and multiple grid sizes |
| OU stationary moments / corrected default width | test-api scientific moments on the default width |
| Low γ, residual, γ=1 rejection, analytical γ=α | test-api scientific regressions; test-failure genuine breakdown snapshots |
| Transfers and callable production | test-api resources and drift; downstream integration real interpolant and floor |
| Wrong lower corner / inward upper boundary | test-kernels boundary checks, fractional γ at negative upper net output |
| Production derivatives / steep SmoothSkiba | test-api marginal production checks; pre-redesign baseline across model families |
| Sampling lengths | test-kernels scalar, vector and multiple-trajectory overloads |
| Derivatives of policy, drift, transition and stationarity | test-api-ad, test-extensions and downstream integration; finite differences plus sparse Hessian |
| Repeatability and initial-value ownership | test-api exact replay, copied warm starts and serialization |
| Caches, sparse/block/Thomas agreement | test-api shared-time/reuse checks; test-kernels structured systems and coupling rejection |
| Save modes / workspace invalidation | Removed public contract: distinct operations and newly constructed Evolution for changed fits |
| Temporal accuracy | test-refinement: two models, two initial distributions, two moments, off-lattice times |
| MATLAB RBC comparison | Historical stencil differs; replaced by OU moments/drift and corrected explicit-grid baseline |
| Unfinished two-asset example | Unsupported API removed |

The historical test count included large per-cell loops. Counts are not a
coverage-equivalence metric. Current tests exercise supported invariants and
public workflows; historical wrappers and intentionally removed behaviors are
not compatibility requirements.

## Archive removal in the deep-module refactor

The former `test/legacy/` directory was removed after checking supported
invariants against the active suite. Its source remains available at commit
`067806e`. `test-migrated-invariants.jl` explicitly covers nonuniform OU and
Poisson density conservation, repeated-solve equality, class permutation,
singletons, and disconnected classes. `test-population-stationary.jl` covers
population defaults, explicit overrides, transient absorption, ambiguous types,
zero-share types, and derivatives of population weights. `test-ownership.jl`
covers read-only result storage, views, detached model snapshots, serialization,
and scalar-read allocation limits.

Tests for arbitrary nonunit measures, public workspace mutation/invalidation,
legacy return arities, and the unfinished two-asset model were intentionally not
ported because those interfaces are unsupported. Historical MATLAB stencil
agreement remains replaced by independent generator/scientific invariants and
the saved explicit-grid fixture, whose historical generator script is retained.
