# Discrete-time implementation and validation

This document records the deterministic milestone. The subsequent ability and
productivity-risk extension is documented in [its validation report](DISCRETE_RISK_VALIDATION.md).

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

Deterministic paths are compared with the analytic steady states, which solve
`1 = discount*(f′(k) + 1 - depreciation)`, to within one grid spacing: both Skiba
branches, and the smooth Ramsey steady state from below and above.

The suite checks budget feasibility, failed-fit behavior, numerical-bound errors,
read-only ownership, warm starts, regridding, resource floors and nested wrappers,
and explicit rejection of unsupported AD inputs. Plotting tests exercise both
new methods through the optional extension.

## Skiba refinement experiment

Run:

```sh
julia --project=. examples/discrete_refinement.jl
julia --project=. examples/discrete_policy_jumps.jl
julia --project=. examples/discrete_continuous_limit.jl
```

The default calibration is discount 0.96, depreciation 0.08, risk aversion 2,
`A_H=0.6`, `A_L=0.4`, `α=0.3`, and `κ=2`. The experiment follows the jump where capital
reverses direction, from `policy_jumps`. This is a calibration-specific result, not
a general claim that every model has one threshold. Paths start 0.05 below and above the estimated switch and run 300
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
grids. The roughly 0.09 jump between competing investment choices at the threshold
persists under refinement, but it is not the only policy jump; see below. Long-run path endpoints are more grid-sensitive than the threshold:
piecewise-linear continuation can pin deterministic steady states to nearby grid
nodes. These results do not justify reporting all path endpoints at the same
precision as the threshold.

A small Bellman residual bounds iteration error for the interpolated problem;
it is not evidence by itself of small state-grid or threshold error. These
experiments support this calibration, not universal error guarantees.

## Absorbing nodes around a smooth steady state

Piecewise-linear continuation can make neighbouring nodes absorbing around one true
steady state. For default Ramsey preferences with `A=1` and bounds `0.05k*` and `1.5k*`, 1,000
requested points give two absorbing nodes, 3.6275 and 3.6327, around `k*=3.6302`.
Paths from below and above end at different nodes, and `stationary` reports multiple
classes. At 500 points there is one. The error stays near half a grid spacing, but
the class structure changes qualitatively with resolution.

This is a correct solution of the interpolated problem. Removing it would need
higher-order interpolation, which loses the interval concavity behind the global
search. Merging nearby classes could hide a real trap, so `stationary` instead
reports each class's capital range. Tests pin this behaviour.

## Policy jump cascade

The escape threshold is one of a sequence of jumps, reported by `policy_jumps`. Run
`julia --project=. examples/discrete_policy_jumps.jl`. At 8,000 requested points:

| Jump at k | Choice below | Choice above | Jump size | Reverses direction |
| ---: | ---: | ---: | ---: | --- |
| 2.8436 | 2.6799 | 2.7153 | 0.0353 | no |
| 2.9582 | 2.8280 | 2.8579 | 0.0299 | no |
| 3.0501 | 2.9452 | 2.9701 | 0.0249 | no |
| 3.1244 | 3.0394 | 3.0601 | 0.0207 | no |
| 3.1847 | 3.1156 | 3.1329 | 0.0173 | no |
| 3.2016 | 3.1482 | 3.2400 | 0.0918 | yes |

Each jump's two choices straddle the previous jump point, and the first straddles
the production crossing at 2.6985. The objective has a convex kink at a convex kink
of the value function, so optimal choices skip it, and the state where the choice
switches sides produces a further kink. Only the last jump changes the direction of
capital; the others occur on the declining path. All six locations agree to within
about 0.0001 from 1,000 to 8,000 points. Generated grids place a knot at the
production crossing but not at these induced kinks.

The cascade need not lie below the threshold. Where the threshold is below the
production crossing, as in the continuous-time comparison below at periods 1 and 0.5,
the other jumps lie above it, on the rising path.

`policy_jumps` omits jumps smaller than their grid cell by default, because a choice
is not resolved more finely than that, but always reports reversals. Without this
exception the default calibration loses its threshold at 250 points and bound 20,
where the jump (0.0798) is just under one cell (0.0803). With `minimum_jump=0`, all 11
deterministic Skiba runs checked had exactly one reversal, down to 100 points, and
smooth Ramsey runs had no jumps. Knot-level sign changes in capital's drift beside a
low steady state have no jump, so they are not reported as thresholds. This is
evidence from these runs, not a proof for every calibration.

## Continuous-time limit

Run `julia --project=. examples/discrete_continuous_limit.jl`. An economy with
period Δ has budget `c + k′ = Δf(k) + (1-δΔ)k` and discount `exp(-ρΔ)`. CRRA
homogeneity makes its policy independent of the constant scale between period
consumption and flow reward, so `c/Δ` should approach the HJB policy. This mapping
is a validation device; the package performs no such conversion.

With ρ = δ = 0.05, the existing HJB solver on 4,000 points has its switch at
2.2777. The flow-consumption error is the largest gap at capital 1.5, 2, 2.5, 4
and 5. The threshold is the reversing jump from `policy_jumps`.

| Period Δ | Capital points | Threshold | Error against HJB | Largest flow-consumption error | Threshold jump |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 2000 | 2.3516 | 0.0739 | 0.0039 | 0.1459 |
| 0.50 | 3000 | 2.3167 | 0.0390 | 0.0021 | 0.0688 |
| 0.25 | 4000 | 2.2979 | 0.0202 | 0.0011 | 0.0342 |
| 0.10 | 5000 | 2.2863 | 0.0085 | 0.0003 | 0.0136 |

The threshold error falls roughly in proportion to Δ, and consumption agrees at the
sampled states. This supports both solvers independently of the analytic benchmark.
It covers this calibration only. The threshold jump also shrinks in proportion to Δ,
which fits the continuous-time switch, where capital's drift changes sign rather than
capital jumping.

No other jump reaches one grid cell at periods 0.25 and 0.1 on these grids, though
periods 1 and 0.5 have four and seven. Grid size changes between these rows, so this
does not establish how the cascade behaves as the period shortens.
`test/test-discrete.jl` checks a coarser version of the limit.

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
julia --project=. examples/discrete_policy_jumps.jl
julia --project=. examples/discrete_continuous_limit.jl
```

The full package suite passes on Julia 1.12.3, including all existing
continuous-time tests and 1,182 discrete-time assertions. README examples, the
capital-path example, and the refinement experiment also execute successfully.
Method-ambiguity detection reports no ambiguities in GrowthModels.

The project retains Julia 1.9 compatibility. No dependency or package-version
change is required. See the PR for additional compatibility and CI results.
