# All grid geometry, coordinates, policy buffers and sparse storage are owned by
# one solve. Returned arrays cannot be overwritten by a later solve or warm start.
struct HJBWorkspace{T,D,C}
    coordinates::C
    y::Array{T,D}
    c0::Array{T,D}
    dkf::Vector{T}
    dkb::Vector{T}
    dVf::Array{T,D}
    dVb::Array{T,D}
    dV0::Array{T,D}
    c::Array{T,D}
    If::BitArray{D}
    Ib::BitArray{D}
    I0::BitArray{D}
    rhs::Vector{T}
    next_v::Array{T,D}
    A::SparseMatrixCSC{T,Int}
    B::SparseMatrixCSC{T,Int}
    diffusion_values::Vector{T}
    diagonal::Vector{Int}
    backward::Vector{Int}
    forward::Vector{Int}
    blocksize::Int
end

model_eltype(::Model{T}) where T = T
concrete_eltype(a) = isconcretetype(eltype(a)) ? eltype(a) : mapreduce(typeof, promote_type, a)

function HJBWorkspace(m::Model, state::StateSpace, hp::StateSpaceHyperParams;
                      flat_transfer = 0.0, diffusion_matrix = nothing, value_type = Float64)
    dims = size(state)
    dims == size(hp) || throw(DimensionMismatch("State grid and hyperparameters must agree"))
    first(keys(state.state)) == :k || throw(ArgumentError("The first state must be capital k"))
    Nk = dims[1]
    Nk >= 2 || throw(ArgumentError("At least two capital grid points are required"))
    diffusion_type = diffusion_matrix === nothing ? Float64 : eltype(diffusion_matrix)
    T = promote_type(Float64, model_eltype(m), concrete_eltype(state.aux_state.y), typeof(flat_transfer), value_type, diffusion_type, map(typeof, Base.values(preferences(m)))...)
    D = length(dims)
    coordinates = NamedTuple{keys(state.state)}(ntuple(j -> repeat(
        reshape(T.(state.state[j]), ntuple(d -> d == j ? dims[j] : 1, D)),
        ntuple(d -> d == j ? 1 : dims[d], D)...), D))
    y = T.(state.aux_state.y .+ flat_transfer)
    size(y) == dims || throw(DimensionMismatch("Production must match the state grid"))
    c0 = y .- preferences(m).δ .* coordinates.k
    all(x -> isfinite(x) && x > 0, selectdim(c0, 1, 1)) ||
        throw(DomainError(c0, "Positive net output is required at every lower-capital boundary state"))
    dkf, dkb = generate_dx(T.(state[:k]))
    all(x -> isfinite(x) && x > 0, dkf) || throw(ArgumentError("Capital grid must be strictly increasing"))
    n = prod(dims)
    diffusion = if diffusion_matrix !== nothing
        SparseMatrixCSC{T,Int}(diffusion_matrix)
    elseif m isa StochasticModel
        SparseMatrixCSC{T,Int}(construct_diffusion_matrix(m.stochasticprocess, state, hp))
    else
        spzeros(T, n, n)
    end
    size(diffusion) == (n, n) || throw(DimensionMismatch("Diffusion matrix must match the state grid"))
    check_generator(diffusion)
    rows, cols, values = findnz(diffusion)
    # Store both capital directions even when their current rates are zero.
    # Updating only nzval below preserves the pattern for symbolic LU reuse.
    for i in 1:n
        push!(rows, i); push!(cols, i); push!(values, zero(T))
        if (i - 1) % Nk != 0
            push!(rows, i); push!(cols, i - 1); push!(values, zero(T))
        end
        if i % Nk != 0
            push!(rows, i); push!(cols, i + 1); push!(values, zero(T))
        end
    end
    A = sparse(rows, cols, values, n, n)
    diagonal, backward, forward = zeros(Int, n), zeros(Int, n), zeros(Int, n)
    for col in 1:n, ptr in nzrange(A, col)
        row = rowvals(A)[ptr]
        if row == col
            diagonal[row] = ptr
        elseif col == row - 1 && (row - 1) % Nk != 0
            backward[row] = ptr
        elseif col == row + 1 && row % Nk != 0
            forward[row] = ptr
        end
    end
    blocksize = D > 2 ? prod(dims[1:2]) : n
    return HJBWorkspace(coordinates, y, c0, dkf, dkb,
        zeros(T, dims), zeros(T, dims), max.(c0, eps()).^(-preferences(m).γ),
        zeros(T, dims), falses(dims), falses(dims), falses(dims),
        zeros(T, n), zeros(T, dims), A, copy(A), copy(nonzeros(A)),
        diagonal, backward, forward, blocksize)
end

function prepare_hjb_step!(ws::HJBWorkspace, m, v, Delta)
    Nk = size(v, 1)
    rates = nonzeros(ws.A)
    copyto!(rates, ws.diffusion_values)
    for offset in 0:Nk:length(v)-1, k in 1:Nk
        i = offset + k
        vf = k < Nk ? (v[i+1] - v[i]) / ws.dkf[k] : ws.dV0[i]
        vb = k > 1 ? (v[i] - v[i-1]) / ws.dkb[k] : ws.dV0[i]
        ws.dVf[i], ws.dVb[i] = vf, vb
        c, If, Ib, I0, sf, sb = upwind_choice(vf, vb, ws.c0[i], preferences(m).γ, k, Nk)
        ws.c[i], ws.If[i], ws.Ib[i], ws.I0[i] = c, If, Ib, I0
        x = Ib ? -sb / ws.dkb[k] : zero(c)
        z = If ? sf / ws.dkf[k] : zero(c)
        rates[ws.diagonal[i]] -= x + z
        k > 1 && (rates[ws.backward[i]] += x)
        k < Nk && (rates[ws.forward[i]] += z)
        ws.rhs[i] = utility(c, preferences(m).γ) + v[i] / Delta
    end
    # The diffusion was checked at construction; selected capital rates are
    # nonnegative by construction and each row adds (-x-z, x, z).
    all(isfinite, rates) && all(isfinite, ws.rhs) ||
        throw(ValueFunctionError("Nonfinite HJB policy or generator"))
    nonzeros(ws.B) .= .-rates
    for i in eachindex(ws.diagonal)
        nonzeros(ws.B)[ws.diagonal[i]] += preferences(m).ρ + 1 / Delta
    end
    return ws
end

function hjb_linear_solver(ws, m, method)
    method in (:auto, :sparse) || throw(ArgumentError("linear_solver must be :auto or :sparse"))
    if method == :auto && m isa DeterministicModel
        return TridiagonalHJBSolver(ws.B)
    end
    return SparseBlockSolver(ws.B; blocksize = method == :auto ? independent_blocksize(ws.B, ws.blocksize) : size(ws.B, 1))
end

convergence_norm(x::Real) = abs(x)
function convergence_norm(x::ForwardDiff.Dual)
    value_norm = convergence_norm(ForwardDiff.value(x))
    tangents = ForwardDiff.partials(x)
    return isempty(tangents) ? value_norm : max(value_norm, maximum(convergence_norm, tangents))
end

function hjb_distance(ws, v)
    distance = 0.0
    for i in eachindex(v)
        distance = max(distance, convergence_norm(ws.next_v[i] - v[i]))
    end
    isfinite(distance) || throw(ValueFunctionError("Nonfinite value or tangent in HJB update"))
    return distance
end

function admissible_hjb_policy(ws)
    return all(i -> isfinite(ws.c[i]) && ws.c[i] > 0 &&
        (!ws.If[i] || ws.dVf[i] > eps()) && (!ws.Ib[i] || ws.dVb[i] > eps()), eachindex(ws.c))
end
