
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

struct HyperParams{T} 
    N::Int64
    xmin::T
    xmax::T
    coef::T
    power::T
    function HyperParams(; N = 1000, xmin = 0.001, xmax = 10.0,  coef = 0.0, power = 0.0)
        new{eltype(xmin)}(N, xmin, xmax, coef, power)
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



function generate_grid(N, xmin, xmax, coef, power)
    x = range(0, 1, length = N)
    uneven_x = x .+ coef * x.^power
    uneven_xmax = maximum(uneven_x)
    uneven_xmin = minimum(uneven_x)
    x_grid = xmin .+ ((xmax - xmin) ./ (uneven_xmax - uneven_xmin)) .* uneven_x
    return x_grid
end


function StateSpace(statespacehyperparams::StateSpaceHyperParams{N, D}, aux_state::NamedTuple) where {N, D}
    names = keys(statespacehyperparams.hyperparams)
    values = map(
        x -> generate_grid(x.N, x.xmin, x.xmax, x.coef, x.power), statespacehyperparams.hyperparams)
    state = NamedTuple(zip(names, values))
    T = typeof(first(values))
    StateSpace{T, N, D, typeof(state), typeof(aux_state)}(state, aux_state)
end


# Stochastic Model + OU Process
function StateSpaceHyperParams(m::StochasticModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40, coef = 0.0, power = 0.0) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    # z_hps
    zmean = process_mean(m.stochasticprocess)
    zmin = zmean*0.8
    zmax = zmean*1.2
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps))
end

# Stochastic Skiba Model + OU Process + Ability
function StateSpaceHyperParams(m::StochasticSkibaAbilityModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 40, Nη = 5, coef = 0.0, power = 0.0) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    kssH = k_steady_state_hi(m)
    # k_hps
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    # z_hps
    zmean = process_mean(m.stochasticprocess)
    zmin = zmean*0.8
    zmax = zmean*1.2
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    # η_hps
    ηmean = m.η_mean
    ηmin = ηmean*0.8
    ηmax = ηmean*1.2
    η_hps = HyperParams(N = Nη, xmax = ηmax, xmin = ηmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
end


# Stochastic Skiba Model + OU Process + Ability
function StateSpaceHyperParams(m::StochasticNPAbilityModel{T, S}; Nk = 1000, kmax = 10.0, kmin = 0.001, Nz = 40, Nη = 5, coef = 0.0, power = 0.0) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    # z_hps
    zmean = process_mean(m.stochasticprocess)
    zmin = zmean*0.8
    zmax = zmean*1.2
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    # η_hps
    ηmean = m.η_mean
    ηmin = ηmean*0.8
    ηmax = ηmean*1.2
    η_hps = HyperParams(N = Nη, xmax = ηmax, xmin = ηmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
end


# Stochastic Model + Poisson Process
function StateSpaceHyperParams(m::StochasticModel{T, S}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, Nz = 2, coef = 0.0, power = 0.0) where {T <: Real, S <: PoissonProcess}
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    # z_hps
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps))
end



function StateSpace(m::StochasticModel{T, S}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    k = generate_grid(k_hps.N, k_hps.xmin, k_hps.xmax, k_hps.coef, k_hps.power)
    z = generate_grid(z_hps.N, z_hps.xmin, z_hps.xmax, z_hps.coef, z_hps.power) 
    # z' creates Nk x Nz matrix
    y = production_function(m, k, z')
    StateSpace((k = k, z = z), (y = y,))
end

function StateSpace(m::StochasticModel{T, S}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: PoissonProcess}
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    k = generate_grid(k_hps.N, k_hps.xmin, k_hps.xmax, k_hps.coef, k_hps.power)
    z = m.stochasticprocess.z
    # z' creates Nk x Nz matrix
    y = production_function(m, k, z')
    StateSpace((k = k, z = z), (y = y,))
end



function StateSpace(m::Union{StochasticNPAbilityModel{T,S},StochasticSkibaAbilityModel{T,S}}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    k_hps = statespacehyperparams[:k]
    z_hps = statespacehyperparams[:z]
    η_hps = statespacehyperparams[:η]
    k = generate_grid(k_hps.N, k_hps.xmin, k_hps.xmax, k_hps.coef, k_hps.power)
    z = generate_grid(z_hps.N, z_hps.xmin, z_hps.xmax, z_hps.coef, z_hps.power)
    η = generate_grid(η_hps.N, η_hps.xmin, η_hps.xmax, η_hps.coef, η_hps.power)
    η_reshape = reshape(η, 1, 1, size(η, 1))
    y = production_function(m, k, z', η_reshape)
    StateSpace((k = k, z = z, η = η), (y = y,))
end


function StateSpaceHyperParams(m::DeterministicModel{T}; Nk = 1000, kmax_f = 1.3, kmin_f = 0.001, coef = 0.0, power = 0.0) where {T <: Real}
    kss = last(k_steady_state(m))
    kmin, kmax = kmin_f*kss, kmax_f*kss
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    return StateSpaceHyperParams((k = k_hps,))
end


function StateSpace(m::DeterministicModel{T}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real}
    k_hps = statespacehyperparams[:k]
    k = generate_grid(k_hps.N, k_hps.xmin, k_hps.xmax, k_hps.coef, k_hps.power)
    y = production_function(m, k)
    StateSpace((k = k,), (y = y,))
end






# Struct to hold value function, its derivatives, and convergence diagnostics
struct Value{T, D} 
    v::Array{T, D}
    dVf::Array{T, D}
    dVb::Array{T, D}
    dV0::Array{T, D}
    A::SparseArrays.SparseMatrixCSC
    dist::Vector{Float64}
    convergence_status::Bool
    iter::Int64
end

function Value(::StateSpace{T, N, D, C }) where {T, N, D, C <: NamedTuple}
    v = zeros(T, D)
    dVf = zeros(T, D)
    dVb = zeros(T, D)
    dV0 = zeros(T, D)
    A_dim = prod(D)
    A = SparseArrays.spzeros(T, (A_dim, A_dim))
    dist = [Inf]
    convergence_status = false
    iter = 0
    Value(v, dVf, dVb, dV0, A, dist, convergence_status, iter)
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
    A_dim = prod(D)
    A = SparseArrays.spzeros(T, (A_dim, A_dim))
    dist = [Inf]
    convergence_status = false
    iter = 0
    Value(v, dVf, dVb, dV0, A, dist, convergence_status, iter)
end

function Value(; v, dVf, dVb, dV0, A, dist, convergence_status, iter) 
    Value(v, dVf, dVb, dV0, A, dist, convergence_status, iter)
end

function Value{T,D}(; v, dVf, dVb, dV0, A, dist, convergence_status, iter) where {T,D}
    Value{T,D}(v, dVf, dVb, dV0, A, dist, convergence_status, iter)
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

"""
    check_statespace_constraints(statespace::GrowthModels.StateSpace, p)

Check if statespace (non-negativity of capital) is satisfied even with 0 consumption. Throw a 
    domain error if not.
"""
function check_statespace_constraints(statespace::StateSpace, p)
    # Check state constraint satisfied and exit early if not 
    max_statespace_constraint = statespace.aux_state.y[end] - p.δ * maximum(statespace[:k])
    min_statespace_constraint = statespace.aux_state.y[1] - p.δ * minimum(statespace[:k])
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
    hyper_params = StateSpaceHyperParams(m)
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
function solve_growth_model(model::Model, init_value::Value, hyper_params::StateSpaceHyperParams; update = false, verbose = true)
    m = model
    check_statespace_constraints(StateSpace(m, hyper_params), m)
    res = solve_HJB(m, hyper_params, init_value = init_value, verbose = verbose)
    if update
        update_value_function!(init_value, res)
    end
    sm = SolvedModel(m, res)
    return sm, res
end
function solve_growth_model(m::Model)
    hyperparams = StateSpaceHyperParams(m)
    state = StateSpace(m, hyperparams)
    init_value = Value(state)
    res = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000, verbose = false)
    sm = SolvedModel(m, res)
    return sm, res
end

function solve_growth_model(m::Model, hp_args)
    hyperparams = StateSpaceHyperParams(m; hp_args...)
    state = StateSpace(m, hyperparams)
    init_value = Value(state)
    res = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000, verbose = false)
    sm = SolvedModel(m, res)
    return sm, res
end





statespace_k_dot(m::Model) =  (variables::NamedTuple) -> variables.y .- m.δ .* variables.k .- variables.c
