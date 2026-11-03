# GrowthModels

[![Build Status](https://github.com/EdJeeOnGitHub/GrowthModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EdJeeOnGitHub/GrowthModels.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Numerical conventions

The HJB generator `A` acts on value functions. Its transpose `A'` advances
**cell probabilities**, not densities. `iterate_g(g, A_t, dx_stacked)` accepts a
density and cell weights, evolves `dx_stacked .* g`, and converts back. It
preserves the input weighted mass without renormalizing. The two-argument
`iterate_g(p, A_t)` instead accepts cell probabilities. `StateEvolution`
normalizes the initial density once.

`StationaryDistribution(A_t, dx_stacked; initial_density = g)` computes the
limiting density associated with `g`, including absorption from transient states.
For disconnected ability groups, their initial probability shares are preserved.
If several closed communicating classes exist, `initial_density` is required;
`StateEvolution` supplies it automatically. With a unique closed class, the
keyword can be omitted.

HJB policy selection compares Hamiltonians when both drift directions are
admissible, choosing one policy for both consumption and the generator. The OU
generator upwinds its drift according to its sign, keeping off-diagonal rates
nonnegative. This changes coarse-grid results relative to the historical MATLAB
RBC fixture, which uses a forward-only drift stencil. That fixture remains a
legacy comparison in the tests; the corrected solver is checked separately.
