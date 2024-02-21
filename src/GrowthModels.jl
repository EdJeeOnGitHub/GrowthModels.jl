module GrowthModels

    # Package dependencies
    using Plots
    using UnicodePlots: lineplot
    using SparseArrays, LinearAlgebra
    import Base: show
    using Printf
    using OrdinaryDiffEq, Interpolations
    using NaNMath: pow
    using ForwardDiff
    import Base:\

    # exports
    ## util exports
    export HyperParams,
           Value,
           StateSpace,
           plot_model, 
           plot_diagnostics,
           ValueFunctionError,
           solve_growth_model
    # solved model output
    export SolvedModel
    export update_v,
           solve_HJB,
           update_value_function!
    ## skiba exports
    export  k_steady_state_hi, k_steady_state_lo, k_star,
           production_function,
           k_dot,
           plot_production_function,
           plot_timepath

    # Models
    export SkibaModel, 
           SmoothSkibaModel,
           RamseyCassKoopmansModel

    # Types
    abstract type Model end



    # Modules
    include("Models.jl")
    include("utils.jl")
    include("HJB.jl")
    include("Skiba.jl")
    include("SmoothSkiba.jl")
    include("RamseyCassKoopmans.jl")
    include("SolvedModel.jl")


    function show(io::IO, h::HyperParams)
        print(io, "HyperParams(N = ", h.N, ", dk = ", @sprintf("%.3g", h.dk), ", kmax = ", @sprintf("%.3g", h.kmax), ", kmin = ", @sprintf("%.3g", h.kmin), ")")
    end

    function show(io::IO, v::Value)
        print(io, "Value function with convergence status: ", v.convergence_status, ". Number of iterations: ", v.iter)
    end



end




