# Discrete-time implementation and validation

## Implemented scope

The deterministic milestones in [the design](DISCRETE_TIME_DESIGN.md) are
implemented: native Ramsey and Skiba constructors, per-period budgets, reversible
investment, best technology each period, log/CRRA utility, global Bellman search,
modified policy iteration, owned solution snapshots, warm starts/regridding,
off-grid maximization, plotting, and reproducible capital-path examples.

The default solver is modified policy iteration with 20 fixed-policy sweeps;
`:value_iteration` is the independent iteration baseline using the same global
Bellman operator. No search pruning or concavity shortcuts are used. Native
models and their resource wrappers route to the discrete implementation; existing
continuous-time resource wrappers retain the HJB route.

Generated grids add known production/floor kinks. Explicit coordinate vectors
remain fixed. A grid's positive lower bound is an artificial saving constraint,
not the economic nonnegativity constraint. Zero-capital/zero-income states are
supported when CRRA utility is finite there, and rejected with a domain explanation
for log utility and risk aversion above one.

Stochastic productivity, population evolution, AD, persistent technology adoption,
irreversible investment, and continuous-time conversion remain outside this PR.

## Scientific checks

`test/test-discrete.jl` checks the global interval search against an independent
20,001-action search with nonconcave, decreasing, and flat continuation values.
It covers truncated feasible intervals, bounds, interior choices, log utility,
and CRRA on both sides of one. A constructed exact tie checks lower-capital
selection and distinct competing-maximum diagnostics.

For log utility, full depreciation, and `f(k)=k^α`, the analytical policy is
`k′=α*discount*k^α`. The tests compare both values and off-grid actions with this
benchmark. They also compare value iteration with modified policy iteration and
verify the converged value against a fixed-policy sparse linear solve. The
transition matrix conserves constants and uses nonnegative interpolation weights.

The suite checks budget feasibility, failed-fit behavior, numerical-bound errors,
read-only ownership, warm starts, regridding, resource floors and nested wrappers,
and explicit rejection of unsupported AD inputs. Plotting tests exercise both
new methods through the optional extension.

## Skiba refinement experiment

Run:

```sh
julia --project=. examples/discrete_refinement.jl
```

The default calibration is discount 0.96, depreciation 0.08, risk aversion 2,
`A_H=0.6`, `A_L=0.4`, `α=0.3`, and `κ=2`. The experiment bisects a verified
negative-to-positive investment switch in `[2.8,3.5]`. This is a calibration-specific
escape-threshold calculation, not a general claim that every model has one
threshold. Paths start 0.05 below and above the estimated switch and run 300
periods, re-optimizing at every continuous capital state.

Results on Julia 1.12.3 (requested grid sizes; generated grids add the crossing):

| Points | Upper bound | Threshold | Left action | Right action | Low path endpoint | High path endpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 20 | 3.21042149 | 3.169584 | 3.249403 | 0.964807 | 3.775912 |
| 500 | 20 | 3.20241150 | 3.148239 | 3.242536 | 1.002954 | 3.728269 |
| 1,000 | 20 | 3.20172293 | 3.147235 | 3.240949 | 0.981932 | 3.744557 |
| 2,000 | 20 | 3.20163028 | 3.148444 | 3.240354 | 0.981441 | 3.752688 |
| 600 | 12 | 3.20176978 | 3.146929 | 3.240695 | 0.982554 | 3.746932 |
| 1,500 | 30 | 3.20169826 | 3.147386 | 3.241074 | 0.981621 | 3.743370 |

The threshold changes by about 0.000093 from 1,000 to 2,000 points. Bounds 12,
20, and 30 at comparable spacing change it by less than 0.00005 relative to the
1,000-point run. A 1,000-point explicit grid without the production crossing
returns the same threshold to the displayed precision in this calibration.

All runs have Bellman residuals below `1.8e-10` and no choices at numerical bounds.
Nevertheless, the 250-point threshold is visibly inaccurate relative to finer
grids. The roughly 0.09 jump between competing investment choices persists under
refinement. Long-run path endpoints are more grid-sensitive than the threshold:
piecewise-linear continuation can pin deterministic steady states to nearby grid
nodes. These results do not justify reporting all path endpoints at the same
precision as the threshold.

A small Bellman residual bounds iteration error for the interpolated problem;
it is not evidence by itself of small state-grid or threshold error. These
experiments support this calibration, not universal error guarantees.

## Performance check

A warmed Julia 1.12.3 run of the default Skiba calibration on a requested
500-point grid took 483 value-iteration sweeps versus 30 modified-policy-iteration
improvements. Observed solve times were 4.37 seconds and 0.36 seconds,
respectively. These are single local measurements, not portable timing guarantees.
Both methods use the same exhaustive interval maximization; automated tests
separately check agreement of their converged values and policies.

## Reproduction

```sh
GKSwstype=100 julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/check-readme.jl
julia --project=. examples/discrete_time.jl
julia --project=. examples/discrete_refinement.jl
```

The full package suite passes on Julia 1.12.3, including all existing
continuous-time tests and 1,080 discrete-time assertions. README examples, the
capital-path example, and the refinement experiment also execute successfully.
Method-ambiguity detection reports no ambiguities in GrowthModels.

The project retains Julia 1.9 compatibility. No dependency or package-version
change is required. See the PR for additional compatibility and CI results.
