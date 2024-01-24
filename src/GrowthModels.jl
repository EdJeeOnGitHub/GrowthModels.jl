module GrowthModels

    # Package dependencies
    using Plots
    using SparseArrays, LinearAlgebra

    # exports
    ## util exports
    export HyperParams,
           Value,
           plot_model, 
           plot_diagnostics
    ## skiba exports
    export SkibaModel, 
           update_v,
           solve_HJB, 
           k_steady_state_hi, k_steady_state_lo, k_star,
           production_function,
           k_dot,
           plot_production_function,
           V_err


    # Types
    abstract type Model end



    # Modules
    include("utils.jl")
    include("Skiba.jl")

end



