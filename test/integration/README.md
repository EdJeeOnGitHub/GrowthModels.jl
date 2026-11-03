# GrowthCounterfactuals integration

This test exercises the real interpolated production object, model income floor,
population shares, solver dispatch, warm starts, value interpolation, and a
stationary-moment derivative against finite differences. It is separate from
the package suite because GrowthCounterfactuals is a downstream project.

For sibling checkouts named `GrowthModels.jl`, `aggregate-transfers`, and
`StochasticEM.jl`, use the checked-in environment (Julia 1.11 or newer):

```sh
julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'
julia --project=test/integration test/integration/growthcounterfactuals.jl
```

Its local Manifest is ignored. This leaves the downstream root Manifest alone.
For other checkout layouts, create an isolated environment explicitly:

```julia
using Pkg
Pkg.activate("/tmp/growth-api-integration")
Pkg.develop(path="/path/to/GrowthModels.jl")
Pkg.develop(path="/path/to/StochasticEM.jl")
Pkg.develop(path="/path/to/aggregate-transfers/packages/GrowthCounterfactuals")
```

Then run from the GrowthModels checkout:

```sh
julia --project=/tmp/growth-api-integration test/integration/growthcounterfactuals.jl
```

The test verifies the resolved GrowthModels source is this checkout. A root
Manifest pinned to an older revision must be updated explicitly before running
counterfactual analysis with the new downstream source.
