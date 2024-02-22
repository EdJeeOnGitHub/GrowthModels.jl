
struct ValueFunctionError <: Exception
    var::String
end
Base.showerror(io::IO, e::ValueFunctionError) = print(io, "ValueFunctionError: Error whilst iterating value function - $(e.var)")


# ForwardDiff of a Dual vector by a sparse matrix
# See: https://github.com/magerton/ForwardDiffSparseSolve.jl
function \(A::SparseMatrixCSC{ForwardDiff.Dual{T, V, N}, P}, b::AbstractVector{G}) where {T, V, N, P<:Integer, G}
    return __FDbackslash(A, b, T, V, N)
end

function __FDbackslash(A, b, T, V, N)

    Areal = ForwardDiff.value.(A)
    breal = ForwardDiff.value.(b)
    outreal = Areal\breal
    M = length(outreal)
    deriv = zeros(V, M, N)
    for i in 1:N
        pAi = ForwardDiff.partials.(A, i)
        pbi = ForwardDiff.partials.(b, i)
        deriv[:, i] = -Areal\(pAi * outreal - pbi)
    end
    out = Vector{eltype(A)}(undef, M)
    for j in eachindex(out)
        out[j] = ForwardDiff.Dual{T}(outreal[j], ForwardDiff.Partials(tuple(deriv[j,:]...)))
    end
    return out
end



# Hyperparameters governing State Space and size of difference we're taking
struct HyperParams
    N::Int64
    dk::Real
    kmax::Real
    kmin::Real
end

# Create a HyperParams object 
# just take kmin and kmax as inputs to create grid
function HyperParams(N = 1000, kmax = 10, kmin = 0.001)
    dk = (kmax-kmin)/(N-1)
    HyperParams(N, dk, kmax, kmin)
end

# Struct to hold value function, its derivatives, and convergence diagnostics
struct Value
    v::Array{Real,1}
    dVf::Array{Real,1}
    dVb::Array{Real,1}
    dV0::Array{Real,1}
    dist::Array{Real,1}
    convergence_status::Bool
    iter::Int64
end

function Value(hyperparams::HyperParams)
    v = zeros(hyperparams.N)
    dVf = zeros(hyperparams.N)
    dVb = zeros(hyperparams.N)
    dV0 = zeros(hyperparams.N)
    dist = fill(Inf, hyperparams.N)
    convergence_status = false
    iter = 0
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end

function Value(; v, dVf, dVb, dV0, dist, convergence_status = false, iter = 0)
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end

# Very simple struct to hold state space, individual models will define a function 
# to construct state space specific to their model
struct StateSpace
    state::NamedTuple
end
# Define a custom getter method
Base.getindex(ss::StateSpace, key::Symbol) = ss.state[key]
# Define a custom property accessor
Base.getproperty(ss::StateSpace, name::Symbol) = getfield(ss, :state)[name]

#### Diagnostic Plotting ####

function plot_diagnostics(m::Model, value::Value, variables::NamedTuple, hyperparams::HyperParams)
    Verr = V_err(m)(value, variables);
    convergence_status, curr_iter = value.convergence_status, value.iter
    subplot = plot(
        layout = (1, 2),
        size = (800, 600)
    ) 
    plot!(
        subplot[1],
        log.(value.dist[1:curr_iter]), 
        label = "\$||V^{n+1} - V^n||\$", 
        xlabel = "Iteration", 
        ylabel = "log(Distance)"
    )
    plot!(
        subplot[2],
        variables.k, 
        Verr, 
        linewidth=2, 
        label="", 
        xlabel="k", 
        ylabel="Error in HJB Equation",
        xlims=(hyperparams.kmin, hyperparams.kmax)
    )
    title!(
        subplot, 
        "Convergence Diagnostics - Status: $(convergence_status)"
        ) 

    return subplot
end

function plot_model(m::Model, value::Value, variables::NamedTuple)
    (; k, y, c) = variables
    (; v, dVf, dVb, dV0, dist) = value
    kstar = k_star(m)
    fit_kdot = k_dot(m)(variables)

    # subplot = plot(layout = (2, 2), size = (800, 600))
    p1 =  plot_production_function(m, collect(k))
    scatter!(p1, [kstar], [production_function(m, kstar)], label="kstar", markersize=4)

    index = findmin(abs.(kstar .- k))[2]
    p2 = plot(k, v, label="V")
    scatter!(p2, [kstar], [v[index]], label="kstar", markersize=4)
    xlabel!(p2, "\$k\$")
    ylabel!(p2, "\$v(k)\$")

    p3 = plot(k, c, label="Consumption, c(k)")
    plot!(p3, k, y .- m.δ .* k, label="Production net of depreciation, f(k) - δk")
    xlabel!(p3, "\$k\$")
    ylabel!(p3, "\$c(k)\$")

    p4 = plot(k, fit_kdot, label="kdot")
    plot!(p4, k, zeros(length(k)), linestyle=:dash, label="zeros")
    scatter!(p4, [kstar], [0], label="kstar", markersize=4)
    xlabel!(p4, "\$k\$")
    ylabel!(p4, "\$s(k)\$")

    subplot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))

    return subplot
end

"""
    check_statespace_constraints(statespace::GrowthModels.StateSpace, p)

Check if statespace (non-negativity of capital) is satisfied even with 0 consumption. Throw a 
    domain error if not.
"""
function check_statespace_constraints(statespace::StateSpace, p)
    # Check state constraint satisfied and exit early if not 
    max_statespace_constraint = statespace.y[end] - p.δ * maximum(statespace.k)
    min_statespace_constraint = statespace.y[1] - p.δ * minimum(statespace.k)
    if max_statespace_constraint < 0 || min_statespace_constraint < 0
        throw(DomainError(p, "State space constraint violated"))
    end
end

"""
    solve_growth_model(model::GrowthModels.Model, init_value::GrowthModels.Value)

Solves the growth model using the specified model and initial value.

# Arguments
- `model::GrowthModels.Model`: The growth model to solve.
- `init_value::GrowthModels.Value`: The initial value for the model.

# Returns
- `sm::SolvedModel`: The solved growth model.

# Example
"""
function solve_growth_model(model::Model, init_value::Value) 
    m = model
    hyper_params = HyperParams(m, N = 1000)
    check_statespace_constraints(StateSpace(m, hyper_params), m)
    res = solve_HJB(m, hyper_params, init_value = init_value)
    # update_value_function!(init_value, res)
    sm = SolvedModel(m, res)
    return sm
end
