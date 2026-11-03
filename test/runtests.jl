using Test, GrowthModels
include("test-api.jl")
include("test-api-ad.jl")
include("test-kernels.jl")
include("test-baseline.jl")
include("test-extensions.jl")
include("test-refinement.jl")
include("test-failure.jl")
include("test-plots.jl")

include("test-refactor-baseline.jl")
include("test-population-stationary.jl")
include("test-ownership.jl")

include("test-migrated-invariants.jl")
include("test-default-bounds.jl")
include("test-residual-precision.jl")
