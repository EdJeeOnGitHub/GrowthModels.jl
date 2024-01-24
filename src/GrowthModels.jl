module GrowthModels

    # Package dependencies
    using Plots
    using SparseArrays, LinearAlgebra
    import Base: show
    using Printf

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
           plot_production_function


    # Types
    abstract type Model end



    # Modules
    include("utils.jl")
    include("Skiba.jl")

    function show(io::IO, m::SkibaModel)
        print(io, "SkibaModel: γ = ", m.γ, ", α = ", m.α, ", ρ = ", m.ρ, ", δ = ", m.δ, ", A_H = ", m.A_H, ", A_L = ", m.A_L, ", κ = ", m.κ)
    end

function show(io::IO, h::HyperParams)
    print(io, "HyperParams(N = ", h.N, ", dk = ", @sprintf("%.3g", h.dk), ", kmax = ", @sprintf("%.3g", h.kmax), ", kmin = ", @sprintf("%.3g", h.kmin), ")")
end

    function show(io::IO, v::Value)
        print(io, "Value function with convergence status: ", v.convergence_status, ". Number of iterations: ", v.iter)
    end
end




