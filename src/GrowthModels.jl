module GrowthModels

    # Package dependencies
    using Plots
    using UnicodePlots: lineplot
    using SparseArrays, LinearAlgebra
    import Base: show
    using Printf
    using Interpolations
    using NaNMath: pow
    using ForwardDiff
    using Random
    import Base:\
    using StatsBase
    import StatsBase: sample
    

    # exports
    ## util exports
    export HyperParams,
           Value,
           StateSpace,
           StateSpaceHyperParams,
           plot_model, 
           plot_diagnostics,
           ValueFunctionError,
           solve_growth_model,
           check_statespace_constraints
    # solved model output
    export SolvedModel
    export update_v,
           solve_HJB,
           update_value_function!
    ## skiba exports
    export  k_steady_state_hi, k_steady_state_lo, k_star,
           production_function,
           k_dot,
           plot_production_function
    export k_steady_state
    # Models
    export SkibaModel, 
           SmoothSkibaModel,
           RamseyCassKoopmansModel,
           StochasticSkibaModel,
           StochasticRamseyCassKoopmansModel,
           StochasticTwoAssetKinkedModel,
           StochasticSkibaAbilityModel,
           StochasticNPAbilityModel
       
    export StateEvolution, iterate_g, create_group_ids
    # stochastic stuff
    export StochasticProcess,
           OrnsteinUhlenbeckProcess,
           from_stationary_OrnsteinUhlenbeckProcess,
           PoissonProcess,
           sample
    # state evolution
    export StateEvolution,
                 create_grid_diag,
                 iterate_g, iterate_g!,
                 StationaryDistribution,
                 weighted_mass,
                 normalize_by_weighted_mass,
                 normalize_by_weighted_mass!




       export Model, DeterministicModel, StochasticModel

    # Types
    abstract type Model{T <: Real} end
    abstract type StochasticProcess end

    abstract type DeterministicModel{T <: Real} <: Model{T} end
    abstract type StochasticModel{T <: Real, S <: StochasticProcess} <: Model{T} end


    # Modules
    include("StochasticProcesses.jl")
    include("Models.jl")
    include("utils.jl")
    include("HJB.jl")
    include("Skiba.jl")
    include("SmoothSkiba.jl")
    include("RamseyCassKoopmans.jl")
    include("StochasticRamseyCassKoopmans.jl")
    include("StochasticSkiba.jl")
    include("StochasticTwoAssetKinked.jl")
    include("StochasticSkibaAbility.jl")
    include("StochasticNPAbility.jl")
    include("SolvedModel.jl")
    include("StateEvolution.jl")


    function show(io::IO, h::HyperParams)
        print(io, "HyperParams(N = ", h.N, ", kmax = ", @sprintf("%.3g", h.xmax), ", kmin = ", @sprintf("%.3g", h.xmin), ")")
    end

    function show(io::IO, v::Value)
        print(io, "Value function with convergence status: ", v.convergence_status, ". Number of iterations: ", v.iter)
    end



end




