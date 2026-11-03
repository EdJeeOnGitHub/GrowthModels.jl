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
    T = promote_type(Float64, model_eltype(m), concrete_eltype(state.aux_state.y), typeof(flat_transfer), value_type)
    D = length(dims)
    coordinates = NamedTuple{keys(state.state)}(ntuple(j -> repeat(
        reshape(T.(state.state[j]), ntuple(d -> d == j ? dims[j] : 1, D)),
        ntuple(d -> d == j ? 1 : dims[d], D)...), D))
    y = T.(state.aux_state.y .+ flat_transfer)
    size(y) == dims || throw(DimensionMismatch("Production must match the state grid"))
    c0 = y .- m.δ .* coordinates.k
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
        zeros(T, dims), zeros(T, dims), max.(c0, eps()).^(-m.γ),
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
        vf = k < Nk ? (v[i+1] - v[i]) / ws.dkf[k] : ws.c0[i]^(-m.γ)
        vb = k > 1 ? (v[i] - v[i-1]) / ws.dkb[k] : ws.c0[i]^(-m.γ)
        ws.dVf[i], ws.dVb[i] = vf, vb
        c, If, Ib, I0, sf, sb = upwind_choice(vf, vb, ws.c0[i], m.γ, k, Nk)
        ws.c[i], ws.If[i], ws.Ib[i], ws.I0[i] = c, If, Ib, I0
        x = Ib ? -sb / ws.dkb[k] : zero(c)
        z = If ? sf / ws.dkf[k] : zero(c)
        rates[ws.diagonal[i]] -= x + z
        k > 1 && (rates[ws.backward[i]] += x)
        k < Nk && (rates[ws.forward[i]] += z)
        ws.rhs[i] = c^(1 - m.γ) / (1 - m.γ) + v[i] / Delta
    end
    # The diffusion was checked at construction; selected capital rates are
    # nonnegative by construction and each row adds (-x-z, x, z).
    all(isfinite, rates) && all(isfinite, ws.rhs) ||
        throw(ValueFunctionError("Nonfinite HJB policy or generator"))
    nonzeros(ws.B) .= .-rates
    for i in eachindex(ws.diagonal)
        nonzeros(ws.B)[ws.diagonal[i]] += m.ρ + 1 / Delta
    end
    return ws
end

function hjb_linear_solver(ws, m, method)
    method in (:auto, :sparse) || throw(ArgumentError("linear_solver must be :auto or :sparse"))
    if method == :auto && m isa DeterministicModel
        return TridiagonalHJBSolver(ws.B)
    end
    return SparseBlockSolver(ws.B; blocksize = method == :auto ? ws.blocksize : size(ws.B, 1))
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

function hjb_value(ws, v, dist, converged, iter)
    return Value(v, ws.dVf, ws.dVb, ws.dV0, ws.A, dist, converged, iter)
end

function hjb_variables(ws)
    return merge((y = ws.y,), ws.coordinates, (c = ws.c, If = ws.If, Ib = ws.Ib, I0 = ws.I0))
end

"""
    solve_HJB(model, hyperparams[, state]; init_value=nothing, maxit=1000,
              crit=1e-6, Delta=1000, linear_solver=:auto, verbose=true,
              flat_transfer=0.0)

Solve using a workspace allocated once per call. An explicit `init_value`
(`Value` or an array) is copied and honored, including an all-zero guess. Only
`nothing` requests the model's default initial guess. Warm-start diagnostics are
reset and the input is never mutated.

`:auto` uses a tridiagonal solve for deterministic models and independent sparse
blocks for fixed abilities. `:sparse` uses one full sparse system, useful for
cross-checks. Numeric sparse factors are refreshed each iteration while the
fixed symbolic pattern is reused. The converged update is returned directly.
"""
function solve_HJB(m::Model, hp::StateSpaceHyperParams, state::StateSpace;
                   init_value = nothing, maxit = 1000, crit = 1e-6, Delta = 1000,
                   verbose = true, flat_transfer = 0.0, linear_solver = :auto)
    maxit isa Integer && maxit >= 0 || throw(ArgumentError("maxit must be a nonnegative integer"))
    crit > 0 && isfinite(crit) || throw(ArgumentError("crit must be finite and positive"))
    Delta > 0 && isfinite(Delta) && m.ρ > 0 || throw(ArgumentError("Delta and the discount rate must be positive"))
    linear_solver in (:auto, :sparse) || throw(ArgumentError("linear_solver must be :auto or :sparse"))
    initial = init_value isa Value ? init_value.v : init_value
    initial === nothing || size(initial) == size(state) || throw(DimensionMismatch("Initial value must match the state grid"))
    value_type = initial === nothing ? Float64 : concrete_eltype(initial)
    ws = HJBWorkspace(m, state, hp; flat_transfer = flat_transfer, value_type = value_type)
    v = similar(ws.next_v)
    v .= initial === nothing ? initial_guess(m, state) : initial
    all(isfinite, v) || throw(ArgumentError("Initial value must be finite"))
    dist = Float64[Inf]
    if maxit == 0
        return (value = hjb_value(ws, v, dist, false, 0), variables = nothing, iter = 0)
    end
    prepare_hjb_step!(ws, m, v, Delta)
    solver = hjb_linear_solver(ws, m, linear_solver)
    for n in 1:maxit
        if n > 1
            prepare_hjb_step!(ws, m, v, Delta)
            refactor!(solver, ws.B)
        end
        solve_linear!(vec(ws.next_v), solver, ws.rhs)
        distance = hjb_distance(ws, v)
        push!(dist, distance)
        if distance < crit
            verbose && println("Value Function Converged, Iteration = ", n)
            return (value = hjb_value(ws, ws.next_v, dist, true, n),
                variables = hjb_variables(ws), iter = n)
        end
        copyto!(v, ws.next_v)
    end
    return (value = hjb_value(ws, v, dist, false, maxit), variables = nothing, iter = maxit)
end

function solve_HJB(m::Model, hp::StateSpaceHyperParams; kwargs...)
    return solve_HJB(m, hp, StateSpace(m, hp); kwargs...)
end

# Compatibility entry points for callers doing individual HJB steps. The main
# solve uses the persistent workspace directly, avoiding setup on every update.
function update_v(m::Model, value::Value, state::StateSpace, hp::StateSpaceHyperParams,
                  diffusion_matrix = nothing; iter = 0, crit = 1e-6, Delta = 1000,
                  verbose = true, flat_transfer = 0.0, linear_solver = :auto)
    Delta > 0 && isfinite(Delta) && m.ρ > 0 || throw(ArgumentError("Delta and the discount rate must be positive"))
    size(value.v) == size(state) || throw(DimensionMismatch("Value must match the state grid"))
    ws = HJBWorkspace(m, state, hp; flat_transfer = flat_transfer,
        diffusion_matrix = diffusion_matrix, value_type = concrete_eltype(value.v))
    prepare_hjb_step!(ws, m, value.v, Delta)
    solver = hjb_linear_solver(ws, m, linear_solver)
    solve_linear!(vec(ws.next_v), solver, ws.rhs)
    distance = hjb_distance(ws, value.v)
    dist = copy(value.dist)
    push!(dist, distance)
    converged = distance < crit
    result = hjb_value(ws, ws.next_v, dist, converged, iter)
    if converged
        verbose && println("Value Function Converged, Iteration = ", iter)
        return result, iter, hjb_variables(ws)
    elseif m isa DeterministicModel
        return result, iter
    else
        capital = copy(ws.A)
        nonzeros(capital) .-= ws.diffusion_values
        return result, iter, capital
    end
end
