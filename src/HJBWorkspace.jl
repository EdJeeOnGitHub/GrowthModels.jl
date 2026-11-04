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

# This layer owns joint generator stacking; grids and processes do not know
# workspace storage or factorization details.
function exogenous_generator(m::Model,g::StateGrid)
    ax=getfield(g,:axes)
    process(m) === nothing && return spzeros(model_eltype(m),length(g),length(g))
    Q=sparse(generator(process(m),ax.z))
    size(Q) == (length(ax.z),length(ax.z)) || throw(DimensionMismatch("Process generator must match its axis"))
    check_generator(Q)
    joint=kron(Q,sparse(I,length(ax.k),length(ax.k)))
    return haskey(ax,:ability) ? kron(sparse(I,length(ax.ability),length(ax.ability)),joint) : joint
end

function HJBWorkspace(m::Model, grid::StateGrid; value_type=Float64)
    dims=size(grid)
    ax=getfield(grid,:axes)
    Nk=dims[1]
    raw_resources=map(i -> resources(m,grid_state(grid,i)),CartesianIndices(dims))
    diffusion_matrix=exogenous_generator(m,grid)
    T=promote_type(Float64,model_eltype(m),concrete_eltype(raw_resources),
        value_type,eltype(diffusion_matrix),map(typeof,Base.values(preferences(m)))...)
    D=length(dims)
    coordinates=NamedTuple{keys(ax)}(ntuple(j -> repeat(
        reshape(T.(ax[j]),ntuple(d -> d == j ? dims[j] : 1,D)),
        ntuple(d -> d == j ? 1 : dims[d],D)...),D))
    y=T.(raw_resources)
    c0=y .- preferences(m).δ .* coordinates.k
    all(x -> isfinite(x) && x > 0,selectdim(c0,1,1)) ||
        throw(DomainError(c0,"Positive net output is required at every lower-capital boundary state"))
    dkf,dkb=generate_dx(T.(ax.k))
    all(x -> isfinite(x) && x > 0,dkf) || throw(ArgumentError("Capital grid must be strictly increasing"))
    n=length(grid)
    diffusion=SparseMatrixCSC{T,Int}(diffusion_matrix)
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
    blocksize = preferred_blocksize(grid)
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
