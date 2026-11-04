# Ability and individual-risk validation

This report accompanies [the implemented design](DISCRETE_RISK_DESIGN.md).
Native model constructors now support fixed ability types, supplied Markov chains,
and Rouwenhorst approximations to log AR(1) productivity. Distribution evolution
uses integer periods. Individual simulations keep capital continuous and sample
the same finite-state shock process.

## Automated checks

`test/test-discrete-risk.jl` verifies:

- Copied/read-only Markov inputs, stochastic rows, positive support, and invalid
  native process parameters.
- Rouwenhorst stationary weights, conditional means and variances, unconditional
  log moments, and negative/high persistence at multiple resolutions.
- Zero-volatility and singleton-chain limits, identity-transition decompositions,
  independent ability solves, and policy invariance to population shares.
- Global stochastic choices against independent dense action enumeration;
  factorized policy action against sparse matrix multiplication and a fixed-policy
  linear solve; modified policy iteration against value iteration.
- State-dependent budgets, resource-floor crossings, categorical query/regridding
  rules, shaped ownership, changed-process warm starts, and private cached
  continuation values shared by querying and simulation.
- Integer-period composition, observation invariance, probability and ability-share
  conservation, streaming moments, invariant probabilities and initial overrides.
- Known transient absorption, multiple closed classes, a period-two chain,
  positive-mass periodicity rejection, and zero-share population handling.
- Simulation timing, budget identities, fixed ability, empirical shock frequencies,
  reproducible RNGs, and continuous-capital policy evaluation.

The deterministic suite remains in place, including its analytical log-growth
benchmark, Skiba threshold refinement, and numerical-failure recovery. The full
package suite also includes the existing continuous-time regressions and optional
plotting, now exercised on multidimensional discrete solutions.

## Capital and shock refinement

Run `julia --project=. examples/discrete_risk_refinement.jl` to reproduce the table.
The calibration uses the default Skiba production and preferences, abilities
`[0.9,1.1]` with equal shares, and log productivity with persistence 0.9, innovation
standard deviation 0.1, and log mean zero. Everyone starts at capital 3.2 and the
middle shock state. The horizon is 40 periods; the target is capital at least 4.
Initial capital and the target are explicitly inserted into each capital grid.

Each Monte Carlo estimate uses 1,000 individuals per ability type, with independent
random streams across types and common seeds across refinements. Standard errors
use the weighted within-type sample variances. Both simulation and grid evolution
use the same finite shock chain: this comparison isolates capital-grid effects
and sampling error, not error relative to continuous Gaussian shocks.

Results on Julia 1.12.3:

| Requested capital points | Shock states | Upper bound | Grid mean capital | Simulation mean ± SE | Grid hit probability | Simulation hit probability ± SE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 5 | 20 | 2.39399 | 2.37627 ± 0.03422 | 0.38637 | 0.3720 ± 0.00950 |
| 200 | 5 | 20 | 2.36924 | 2.34673 ± 0.03397 | 0.37178 | 0.3615 ± 0.00944 |
| 400 | 5 | 20 | 2.36351 | 2.33301 ± 0.03393 | 0.36671 | 0.3600 ± 0.00944 |
| 200 | 3 | 20 | 2.34718 | 2.29238 ± 0.03279 | 0.34778 | 0.3400 ± 0.00953 |
| 200 | 7 | 20 | 2.38511 | 2.37357 ± 0.03321 | 0.37689 | 0.3790 ± 0.00941 |
| 300 | 5 | 30 | 2.37024 | 2.34654 ± 0.03397 | 0.37238 | 0.3625 ± 0.00945 |

For the mean capital and hitting probabilities in this table, grid and
continuous-capital simulations differ by less than two estimated standard errors. This is a finite-sample consistency check, not a
guarantee that interpolation error is negligible. Capital refinement still
changes the computed hitting probability, and shock refinement from three to seven
states shifts it by about 0.029. Expanding bounds at similar capital spacing
changes the five-state hitting probability by about 0.0006.

All solves have global Bellman residual below `2e-10`. Each joint chain has two
closed classes in this calibration (one per ability). Invariant mean capital
changes from 2.06326 to 2.03234 to 2.01971 as capital resolution increases from
100 to 200 to 400 at five shock states: long-run moments need further refinement
before high-precision scientific use.

Finite Rouwenhorst support can create different rare-event behavior and class
topology from an unbounded Gaussian process. The moment tests and this finite-chain
simulation comparison do not certify Gaussian tail or escape probabilities.

## Persistence and transition times

The experiment also compares persistence while holding stationary log variance
fixed at `0.1²/(1-0.9²)`. Innovation standard deviation changes accordingly. At
five shock states the stationary shock distribution and its level support are
therefore unchanged; differences reflect persistence and its interaction with
saving behavior, rather than increased unconditional shock dispersion.

The restricted waiting time is `E[min(τ,40)]`, where `τ` is the first period in
which capital reaches 4. Individuals who do not hit within the horizon contribute
40. It is not an assertion that unconditional expected hitting time is finite.

| Persistence | Innovation SD | Grid hit probability | Simulation hit probability ± SE | Grid restricted wait | Simulation restricted wait ± SE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.198680 | 0.43357 | 0.4345 ± 0.00666 | 28.52936 | 28.7795 ± 0.21476 |
| 0.90 | 0.100000 | 0.37178 | 0.3615 ± 0.00944 | 30.97348 | 31.3665 ± 0.25219 |
| 0.95 | 0.071635 | 0.32195 | 0.3090 ± 0.00913 | 32.82108 | 33.1165 ± 0.22755 |

These use 200 requested capital points and bound 20. In this calibration and
initial condition, greater persistence lowers the 40-period target-hitting
probability and increases restricted waiting time. This is not a general
comparative-statics result for all Skiba calibrations.

Waiting times reveal additional capital-grid sensitivity: at 100 points and
persistence 0.9, the grid estimate is 30.40444 versus simulation 31.0885 ± 0.25470
(about 2.7 standard errors apart). At 400 points the estimates are 31.15760 and
31.4435 ± 0.25145 (about 1.1 standard errors apart). Agreement of simpler moments
alone would not detect this sensitivity.

## Operator storage and performance

At five shock states and two types, requested capital resolutions 100, 200, and
400 produce 9,920, 19,625, and 38,900 nonzeros in the policy matrix, respectively.
Measured matrix storage is approximately 167 KB, 330 KB, and 655 KB. The seven-state,
200-point case uses 38,570 nonzeros and about 640 KB. The stored matrix is sparse,
but dense exogenous transitions imply quadratic growth in shock-state count.

Fixed-policy sweeps apply the factorized operator and construct no joint matrix.
The final sparse matrix is materialized once for diagnostics and evolution.
Observed solve times after compilation were roughly 0.51 seconds for 200 capital
points/five shocks and 1.94 seconds for 400 points/five shocks. These are local
measurements, not performance guarantees.

## Reproduction commands

```sh
GKSwstype=100 julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/check-readme.jl
julia --project=. examples/discrete_risk.jl
julia --project=. examples/discrete_risk_refinement.jl
```

The full package suites pass on Julia 1.12.3 and 1.9.4, including the original
continuous-time regressions, 1,080 deterministic discrete-time assertions, 334
new stochastic/ability assertions, and the plotting extension. README examples,
the population/individual-path example, and the refinement experiment execute
successfully. The implementation adds no dependencies. See the PR for CI results.
