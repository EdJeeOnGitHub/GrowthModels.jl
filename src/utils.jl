
struct ValueFunctionError <: Exception
    var::String
end
Base.showerror(io::IO, e::ValueFunctionError) = print(io, "ValueFunctionError: Error whilst iterating value function - $(e.var)")


# Recursive factorization keeps one primal LU even for nested dual numbers.
# Each tangent solves A*ẋ = ḃ - Ȧ*x with that same factorization.
struct DifferentiableSparseFactor{A,P,F}
    matrix::A
    primal_matrix::P
    primal_factor::F
end

factor_sparse(A::SparseMatrixCSC) = lu(A)
function factor_sparse(A::SparseMatrixCSC{<:ForwardDiff.Dual})
    primal = SparseMatrixCSC(size(A, 1), size(A, 2), copy(A.colptr), copy(rowvals(A)),
        ForwardDiff.value.(nonzeros(A)))
    return DifferentiableSparseFactor(A, primal, factor_sparse(primal))
end

refactor_sparse!(factor, A) = lu!(factor, A; reuse_symbolic = true)
function refactor_sparse!(factor::DifferentiableSparseFactor, A)
    A.colptr == factor.primal_matrix.colptr && rowvals(A) == rowvals(factor.primal_matrix) ||
        throw(ArgumentError("Differentiable factorization requires a fixed sparsity pattern"))
    copyto!(nonzeros(factor.matrix), nonzeros(A))
    nonzeros(factor.primal_matrix) .= ForwardDiff.value.(nonzeros(A))
    refactor_sparse!(factor.primal_factor, factor.primal_matrix)
    return factor
end

function solve_sparse_factor(factor, b)
    if eltype(b) <: ForwardDiff.Dual
        return solve_dual_rhs(factor, b)
    end
    return factor \ b
end
function solve_dual_rhs(factor, b::AbstractVector{ForwardDiff.Dual{Tag,V,N}}) where {Tag,V,N}
    real = solve_sparse_factor(factor, ForwardDiff.value.(b))
    partials = ntuple(i -> solve_sparse_factor(factor, [ForwardDiff.partials(x,i) for x in b]), N)
    return [ForwardDiff.Dual{Tag}(real[j], ForwardDiff.Partials(ntuple(i -> partials[i][j],N))) for j in eachindex(b)]
end
function solve_sparse_factor!(out, factor, b)
    if eltype(b) <: ForwardDiff.Dual
        return copyto!(out,solve_sparse_factor(factor,b))
    end
    return ldiv!(out,factor,b)
end
function solve_sparse_factor!(out, factor::DifferentiableSparseFactor, b)
    return copyto!(out, solve_sparse_factor(factor, b))
end

function solve_sparse_factor(factor::DifferentiableSparseFactor{<:SparseMatrixCSC{ForwardDiff.Dual{Tag,V,N}}}, b) where {Tag,V,N}
    A = factor.matrix
    outreal = solve_sparse_factor(factor.primal_factor, ForwardDiff.value.(b))
    M = length(outreal)
    deriv = zeros(V, M, N)
    # Assemble tangents directly from CSC entries, without N derivative matrices.
    for i in 1:N, j in 1:M
        deriv[j, i] = ForwardDiff.partials(b[j], i)
    end
    for col in axes(A, 2), ptr in nzrange(A, col)
        row = rowvals(A)[ptr]
        a = nonzeros(A)[ptr]
        for i in 1:N
            deriv[row, i] -= ForwardDiff.partials(a, i) * outreal[col]
        end
    end
    scratch = similar(outreal)
    for i in 1:N
        column = view(deriv, :, i)
        solve_sparse_factor!(scratch, factor.primal_factor, column)
        copyto!(column, scratch)
    end
    out = Vector{eltype(A)}(undef, M)
    for j in eachindex(out)
        out[j] = ForwardDiff.Dual{Tag}(outreal[j], ForwardDiff.Partials(ntuple(i -> deriv[j, i], N)))
    end
    return out
end

function __FDbackslash(A, b, T, V, N)
    return solve_sparse_factor(factor_sparse(A), b)
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
function StateSpace(state::NamedTuple, aux_state::NamedTuple)
    T = promote_type(map(eltype, values(state))..., map(eltype, values(aux_state))...)
    coordinates = map(x -> T.(x), state)
    auxiliary = map(x -> T.(x), aux_state)
    N = length(coordinates)
    return StateSpace{T, N, tuple_size(coordinates), typeof(coordinates), typeof(auxiliary)}(coordinates, auxiliary)
end
function Base.eltype(::StateSpace{T,N,D,C}) where {T,N,D,C}
    NamedTuple{fieldnames(C), NTuple{N,T}}
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
    N >= 2 || throw(ArgumentError("At least two grid points are required"))
    x = range(0, 1, length = N)
    # Avoid evaluating 0^Dual(power): its generic derivative contains log(0),
    # even when the exponent has zero tangents. At positive powers the endpoint
    # is identically zero; a constant zero power is identically one.
    uneven_x = if iszero(coef)
        x
    else
        powers = [iszero(t) && power > 0 ? zero(power) :
                  iszero(power) ? one(power) : t^power for t in x]
        x .+ coef .* powers
    end
    lo, hi = extrema(uneven_x)
    return collect(xmin .+ (xmax - xmin) .* ((uneven_x .- lo) ./ (hi - lo)))
end


function StateSpace(statespacehyperparams::StateSpaceHyperParams{N, D}, aux_state::NamedTuple) where {N, D}
    names = keys(statespacehyperparams.hyperparams)
    values = map(
        x -> generate_grid(x.N, x.xmin, x.xmax, x.coef, x.power), statespacehyperparams.hyperparams)
    state = NamedTuple(zip(names, values))
    return StateSpace(state, aux_state)
end


# Bounds for log productivity use the stationary log standard deviation.
function productivity_bounds(p::OrnsteinUhlenbeckProcess, standard_deviations = 4)
    p.θ > 0 && p.σ >= 0 || throw(ArgumentError("OU grid requires positive mean reversion and nonnegative volatility"))
    standard_deviations > 0 || throw(ArgumentError("z_width must be positive"))
    width = standard_deviations * p.σ / sqrt(2p.θ)
    return exp(-width), exp(width)
end

function capital_reference(m::StochasticModel, z, η = one(z))
    shock = m.stochasticprocess isa OrnsteinUhlenbeckProcess ? z : one(z)
    if m isa StochasticRamseyCassKoopmansModel
        return (m.α * m.A * shock / (m.ρ + m.δ))^(1 / (1 - m.α))
    end
    return max((m.α * m.A_H * shock * η / (m.ρ + m.δ))^(1 / (1 - m.α)) + m.κ,
        (m.α * m.A_L * shock * η / (m.ρ + m.δ))^(1 / (1 - m.α)))
end

function StateSpaceHyperParams(m::StochasticModel{T,S}; Nk = 1000, kmax_f = 1.3,
        kmin_f = 0.001, Nz = 40, coef = 0.0, power = 0.0, z_width = 4) where {T <: Real,S <: OrnsteinUhlenbeckProcess}
    zmin, zmax = productivity_bounds(m.stochasticprocess, z_width)
    k_hps = HyperParams(N = Nk, xmin = kmin_f * k_steady_state_hi(m),
        xmax = kmax_f * capital_reference(m, zmax), coef = coef, power = power)
    return StateSpaceHyperParams((k = k_hps, z = HyperParams(N = Nz, xmin = zmin, xmax = zmax)))
end

function StateSpaceHyperParams(m::StochasticSkibaAbilityModel{T,S}; Nk = 1000, kmax_f = 1.3,
        kmin_f = 0.001, Nz = 40, Nη = 5, coef = 0.0, power = 0.0, z_width = 4) where {T <: Real,S <: OrnsteinUhlenbeckProcess}
    zmin, zmax = productivity_bounds(m.stochasticprocess, z_width)
    ηmin, ηmax = m.η_mean * 0.8, m.η_mean * 1.2
    k_hps = HyperParams(N = Nk, xmin = kmin_f * k_steady_state_hi(m),
        xmax = kmax_f * capital_reference(m, zmax, ηmax), coef = coef, power = power)
    return StateSpaceHyperParams((k = k_hps, z = HyperParams(N = Nz, xmin = zmin, xmax = zmax),
        η = HyperParams(N = Nη, xmin = ηmin, xmax = ηmax)))
end

function StateSpaceHyperParams(m::StochasticNPAbilityModel{T, S}; Nk = 1000, kmax = 10.0, kmin = 0.001, Nz = 40, Nη = 5, coef = 0.0, power = 0.0, z_width = 4) where {T <: Real, S <: OrnsteinUhlenbeckProcess}
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    # z_hps
    zmin, zmax = productivity_bounds(m.stochasticprocess, z_width)
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    # η_hps
    ηmean = m.η_mean
    ηmin = ηmean*0.8
    ηmax = ηmean*1.2
    η_hps = HyperParams(N = Nη, xmax = ηmax, xmin = ηmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
end

function StateSpaceHyperParams(m::StochasticNPAbilityModel{T, S}; Nk = 1000, kmax = 10.0, kmin = 0.001, Nη = 5, coef = 0.0, power = 0.0) where {T <: Real, S <: PoissonProcess}
    k_hps = HyperParams(N = Nk, xmax = kmax, xmin = kmin, coef = coef, power = power)
    # z_hps
    z = m.stochasticprocess.z
    zmin = z[1]
    zmax = z[end]
    Nz = length(z)
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    # η_hps
    ηmean = m.η_mean
    ηmin = ηmean*0.8
    ηmax = ηmean*1.2
    η_hps = HyperParams(N = Nη, xmax = ηmax, xmin = ηmin)
    return StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
end


# Poisson state counts and support belong to the process.
function StateSpaceHyperParams(m::StochasticModel{T,S}; Nk = 1000, kmax_f = 1.3,
        kmin_f = 0.001, Nz = length(m.stochasticprocess.z), coef = 0.0, power = 0.0,
        Nη = 5) where {T <: Real,S <: PoissonProcess}
    z = m.stochasticprocess.z
    Nz == length(z) || throw(DimensionMismatch("Nz must match the Poisson process support"))
    ability = m isa StochasticSkibaAbilityModel
    ηmax = ability ? m.η_mean * 1.2 : one(eltype(z))
    k_hps = HyperParams(N = Nk, xmin = kmin_f * k_steady_state_hi(m),
        xmax = kmax_f * capital_reference(m, maximum(z), ηmax), coef = coef, power = power)
    base = (k = k_hps, z = HyperParams(N = Nz, xmin = minimum(z), xmax = maximum(z)))
    return StateSpaceHyperParams(ability ? merge(base,
        (η = HyperParams(N = Nη, xmin = m.η_mean * 0.8, xmax = ηmax),)) : base)
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


function StateSpace(m::Union{StochasticNPAbilityModel{T,S},StochasticSkibaAbilityModel{T,S}}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: PoissonProcess}
    k_hps = statespacehyperparams[:k]
    η_hps = statespacehyperparams[:η]
    k = generate_grid(k_hps.N, k_hps.xmin, k_hps.xmax, k_hps.coef, k_hps.power)
    z = m.stochasticprocess.z
    η = generate_grid(η_hps.N, η_hps.xmin, η_hps.xmax, η_hps.coef, η_hps.power)
    η_reshape = reshape(η, 1, 1, size(η, 1))
    y = production_function(m, k, z', η_reshape)
    StateSpace((k = k, z = z, η = η), (y = y,))
end

function StateSpace(m::Union{StochasticNPAbilityModel{T,S},StochasticSkibaAbilityModel{T,S}}, statespacehyperparams::StateSpaceHyperParams) where {T <: Real, S <: StochasticProcess}
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


# Allow passing eta directly
function StateSpace(m::Union{StochasticNPAbilityModel{T,S},StochasticSkibaAbilityModel{T,S}}, k_hps::HyperParams, η) where {T <: Real, S <: PoissonProcess}
    k = generate_grid(k_hps.N, k_hps.xmin, k_hps.xmax, k_hps.coef, k_hps.power)
    z = m.stochasticprocess.z
    η_reshape = reshape(η, 1, 1, size(η, 1))
    y = production_function(m, k, z', η_reshape)
    StateSpace((k = k, z = z, η = η), (y = y,))
end



# Struct to hold value function, its derivatives, and convergence diagnostics
