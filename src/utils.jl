
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



# modified StateSpace construct inspired by Mathieu Gomez
# ensures each state variable is a vector with same dimension
# probably not a great modification but makes things easier for me 
# when I create a struct to hold value functions
struct StateSpace{T, N, D, C <: NamedTuple, A <: NamedTuple} 
    state::C
    aux_state::A
end

Base.ndims(::StateSpace{T, N}) where {T, N} = N
Base.size(statespace::StateSpace{T, N}) where {T, N} = ntuple(i -> length(statespace.state[i]), N)

tuple_size(x) = ntuple(i -> length(x[i]), length(x))
function StateSpace(state::NamedTuple{Names, <: NTuple{N, <: AbstractVector{T}}}, aux_state::NamedTuple{Names_a, <: NTuple{N_a, <: AbstractArray{T}}}) where {Names, N, T, Names_a, N_a}
    StateSpace{T, N, tuple_size(state), typeof(state), typeof(aux_state)}(state, aux_state)
end
function Base.eltype(::StateSpace{T, N, <: NamedTuple{Names, V}}) where {T, N, Names, V}
    NamedTuple{Names, NTuple{N, T}}
end


# Adjust the indexing functions if necessary to work with the new struct definition.
Base.eachindex(statespace::StateSpace) = CartesianIndices(size(statespace))
function Base.getindex(statespace::StateSpace, args::CartesianIndex)
    eltype(statespace)(ntuple(i -> statespace.state[i][args[i]], ndims(statespace)))
end
Base.getindex(statespace::StateSpace, x::Symbol) = statespace.state[x]

struct HyperParams
    N::Int64
    dx::Real
    xmin::Real
    xmax::Real
    function HyperParams(; N = 1000, xmin = 0.001, xmax = 10.0)
        dx = (xmax - xmin) / (N - 1) 
        new(N, dx, xmin, xmax)
    end
end

struct StateSpaceHyperParams{N, D}
    hyperparams::NamedTuple
end

function StateSpaceHyperParams(hyperparams::NamedTuple{Names, <: NTuple{N, <: HyperParams}}) where {Names, N}
    D = ntuple(i -> hyperparams[i].N, N)
    StateSpaceHyperParams{N,D}(hyperparams)
end
Base.getindex(statespacehyperparams::StateSpaceHyperParams, x::Symbol) = statespacehyperparams.hyperparams[x]
Base.size(::StateSpaceHyperParams{N, D}) where {N, D} = D
Base.size(h::HyperParams) = (h.N,)






function StateSpace(statespacehyperparams::StateSpaceHyperParams{N, D}, aux_state::NamedTuple) where {N, D}
    names = keys(statespacehyperparams.hyperparams)
    values = map(
        x -> collect(
            range(x.xmin, x.xmax, length = x.N)
        ), statespacehyperparams.hyperparams)
    state = NamedTuple(zip(names, values))
    T = typeof(first(values))
    StateSpace{T, N, D, typeof(state), typeof(aux_state)}(state, aux_state)
end



# Struct to hold value function, its derivatives, and convergence diagnostics
struct Value{T, D} 
    v::Array{T, D}
    dVf::Array{T, D}
    dVb::Array{T, D}
    dV0::Array{T, D}
    dist::Vector{Float64}
    convergence_status::Bool
    iter::Int64
end

function Value(::StateSpace{T, N, D, C }) where {T, N, D, C <: NamedTuple}
    v = zeros(T, D)
    dVf = zeros(T, D)
    dVb = zeros(T, D)
    dV0 = zeros(T, D)
    dist = [Inf]
    convergence_status = false
    iter = 0
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end

function Value(T, h::Union{HyperParams,StateSpaceHyperParams{N,D}}) where {N, D}
    D_v = isa(h, HyperParams) ? h.N : D
    Value(T, D_v)
end

function Value(T, D)
    v = zeros(T, D)
    dVf = zeros(T, D)
    dVb = zeros(T, D)
    dV0 = zeros(T, D)
    dist = [Inf]
    convergence_status = false
    iter = 0
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end

function Value(; v, dVf, dVb, dV0, dist, convergence_status, iter) 
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end

function Value{T,D}(; v, dVf, dVb, dV0, dist, convergence_status, iter) where {T,D}
    Value{T,D}(v, dVf, dVb, dV0, dist, convergence_status, iter)
end


#### Diagnostic Plotting ####

function plot_diagnostics(m::Model, value::Value, variables::NamedTuple, hyperparams::StateSpaceHyperParams)
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
        xlims=(hyperparams[:k].xmin, hyperparams[:k].xmax)
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
    fit_kdot = statespace_k_dot(m)(variables)

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
function solve_growth_model(model::Model, init_value::Value; update = false, verbose = true) 
    m = model
    hyper_params = HyperParams(m, N = 1000)
    check_statespace_constraints(StateSpace(m, hyper_params), m)
    res = solve_HJB(m, hyper_params, init_value = init_value, verbose = verbose)
    if update
        update_value_function!(init_value, res)
    end
    sm = SolvedModel(m, res)
    return sm, res
end


"""
    solve_growth_model(model::Model, init_value::Value, hyper_params::HyperParams; update = false)

Solves the growth model using the specified model, initial value, and hyperparameters.

# Arguments
- `model::Model`: The growth model to solve.
- `init_value::Value`: The initial value for the value function.
- `hyper_params::HyperParams`: The hyperparameters for the growth model.
- `update::Bool`: (optional) Whether to update the initial value function with the result. Default is `false`.

# Returns
- `sm::SolvedModel`: The solved growth model.
- `res::Result`: The result of solving the growth model.

"""
function solve_growth_model(model::Model, init_value::Value, hyper_params::HyperParams; update = false, verbose = true)
    m = model
    check_statespace_constraints(StateSpace(m, hyper_params), m)
    res = solve_HJB(m, hyper_params, init_value = init_value, verbose = verbose)
    if update
        update_value_function!(init_value, res)
    end
    sm = SolvedModel(m, res)
    return sm, res
end


statespace_k_dot(m::Model) =  (variables::NamedTuple) -> variables.y .- m.δ .* variables.k .- variables.c