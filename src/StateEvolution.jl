struct StateEvolution{T <: Real}
    S::Matrix{T}
    times::Vector{Float64}
    g::Vector{T}
    A::SparseMatrixCSC{T,Int}
    dx_stacked::Vector{T}
    cell_masses::Array{T}
    stationary_dist::Union{Nothing,Vector{T}}
    final_density::Vector{T}
    final_time::Float64
end

function Base.getindex(s::StateEvolution, t::Real)
    index = searchsortedfirst(s.times, t)
    index <= length(s.times) && s.times[index] == t ||
        throw(ArgumentError("Time $t was not saved"))
    return s.S[:, index]
end

# Implicit Euler method
function iterate_g!(g, A_t; time_step = 1)
    evolution = I - time_step * A_t
    g .= evolution \ g
end
# Implicit Euler method
function iterate_g(g, A_t; time_step = 1)
    evolution = I - time_step * A_t
    return evolution \ g
end

"""
    iterate_g(g, A_t, dx_stacked; time_step = 1)

Advance a density using the transpose of the HJB generator. Internally evolve
cell masses `dx_stacked .* g`, then convert back to a density. This preserves
weighted mass without renormalization, including on nonuniform grids.
The two-argument overload instead advances cell masses directly.
"""
function iterate_g(g, A_t, dx_stacked; time_step = 1, workspace = nothing)
    if workspace !== nothing
        update_evolution!(workspace, A_t; time_step = time_step, dx_stacked = dx_stacked)
        return iterate_g(g, workspace)
    end
    evolution = I - time_step * A_t
    return (evolution \ (dx_stacked .* g)) ./ dx_stacked
end

function iterate_g!(g, A_t, dx_stacked; time_step = 1, workspace = nothing)
    if workspace !== nothing
        update_evolution!(workspace, A_t; time_step = time_step, dx_stacked = dx_stacked)
        return iterate_g!(g, workspace)
    end
    g .= iterate_g(g, A_t, dx_stacked; time_step = time_step)
    return g
end

function create_dx_stacked(x, N_second_dim)
    dxf, dxb = generate_dx(x)
    dx_tilde = 0.5 * (dxf + dxb)
    dx_tilde[1] = 0.5*dxf[1]
    dx_tilde[end] = 0.5*dxb[end]
    dx_tilde_stacked = repeat(dx_tilde, N_second_dim)
    return dx_tilde_stacked
end

# Utility functions for mass conservation
"""
    weighted_mass(g, dx_stacked)

Compute the weighted mass of distribution g using grid spacing from dx_stacked.
This is the correct way to measure total probability on non-uniform grids.
"""
function weighted_mass(g, dx_stacked)
    return sum(dx_stacked .* g)
end

"""
    normalize_by_weighted_mass!(g, dx_stacked)

Normalize distribution g by its weighted mass to ensure proper mass conservation.
This modifies g in place.
"""
function normalize_by_weighted_mass!(g, dx_stacked)
    mass = weighted_mass(g, dx_stacked)
    g ./= mass
    return g
end

"""
    normalize_by_weighted_mass(g, dx_stacked)

Normalize distribution g by its weighted mass to ensure proper mass conservation.
This returns a new normalized distribution.
"""
function normalize_by_weighted_mass(g, dx_stacked)
    mass = weighted_mass(g, dx_stacked)
    return g ./ mass
end

# Strongly connected components of the transition graph (Kosaraju).
# Columns of A_t are source states. Iterative DFS avoids recursion limits on
# large grids; only strictly positive off-diagonal entries are graph edges.
function closed_communicating_classes(A_t::SparseMatrixCSC)
    n = size(A_t, 1)
    seen = falses(n)
    order = Int[]
    rows, vals = rowvals(A_t), nonzeros(A_t)
    stack = Tuple{Int, Bool}[]
    for root in 1:n
        seen[root] && continue
        push!(stack, (root, false))
        while !isempty(stack)
            v, finished = pop!(stack)
            if finished
                push!(order, v)
            elseif !seen[v]
                seen[v] = true
                push!(stack, (v, true))
                for ptr in nzrange(A_t, v)
                    w = rows[ptr]
                    if w != v && vals[ptr] > 0 && !seen[w]
                        push!(stack, (w, false))
                    end
                end
            end
        end
    end

    reverse_graph = sparse(transpose(A_t))
    reverse_rows, reverse_vals = rowvals(reverse_graph), nonzeros(reverse_graph)
    labels = zeros(Int, n)
    components = Vector{Int}[]
    pending = Int[]
    for root in Iterators.reverse(order)
        labels[root] != 0 && continue
        push!(components, Int[])
        label = length(components)
        labels[root] = label
        push!(pending, root)
        while !isempty(pending)
            v = pop!(pending)
            push!(components[label], v)
            for ptr in nzrange(reverse_graph, v)
                w = reverse_rows[ptr]
                if w != v && reverse_vals[ptr] > 0 && labels[w] == 0
                    labels[w] = label
                    push!(pending, w)
                end
            end
        end
    end

    closed = trues(length(components))
    for v in 1:n, ptr in nzrange(A_t, v)
        if vals[ptr] > 0 && labels[rows[ptr]] != labels[v]
            closed[labels[v]] = false
        end
    end
    return components[closed], findall(v -> !closed[labels[v]], 1:n)
end

"""
    StationaryDistribution(A_t, dx_stacked; initial_density = nothing)

Return a stationary density for the transpose of an HJB generator. If there
are multiple closed communicating classes, supply `initial_density` to select
its long-run mixture. Initial mass in transient states is allocated using
absorption probabilities; disconnected ability groups retain their mass.
Without an initial density, a unique closed class is required.
"""
function StationaryDistribution(A_t::SparseMatrixCSC, dx_stacked;
                                initial_density = nothing)
    check_generator(sparse(transpose(A_t)))
    n = size(A_t, 1)
    length(dx_stacked) == n || throw(DimensionMismatch("One cell weight is required per state"))
    all(w -> isfinite(w) && w > 0, dx_stacked) ||
        throw(ArgumentError("Cell weights must be finite and positive"))
    classes, transient = closed_communicating_classes(A_t)
    if isnothing(initial_density)
        length(classes) == 1 || throw(ArgumentError(
            "Multiple stationary classes: supply initial_density to determine their masses"))
        initial_mass = nothing
    else
        length(initial_density) == n || throw(DimensionMismatch("Initial density must match the generator"))
        all(g -> isfinite(g) && g >= 0, initial_density) ||
            throw(ArgumentError("Initial density must be finite and nonnegative"))
        initial_mass = float.(dx_stacked .* initial_density)
        sum(initial_mass) > 0 || throw(ArgumentError("Initial density must have positive mass"))
        initial_mass ./= sum(initial_mass)
    end

    # Integrated transient occupancy solves -A_TT * occupation = initial_mass_T.
    # Its outflow into each closed class is that class's eventual absorbed mass.
    occupation = if initial_mass !== nothing && !isempty(transient)
        (-A_t[transient, transient]) \ initial_mass[transient]
    else
        nothing
    end
    mass = zeros(promote_type(Float64, eltype(A_t), eltype(dx_stacked)), n)
    for states in classes
        weight = if initial_mass === nothing
            one(eltype(mass))
        else
            sum(initial_mass[states]) + (occupation === nothing ? zero(eltype(mass)) :
                sum(A_t[states, transient] * occupation))
        end
        # An irreducible closed block has a one-dimensional nullspace. Pin one
        # component, then normalize within the class before applying its weight.
        block = copy(A_t[states, states])
        block[1, :] .= 0
        block[1, 1] = 1
        rhs = zeros(eltype(mass), length(states))
        rhs[1] = 1
        stationary_mass = block \ rhs
        mass[states] .= weight .* stationary_mass ./ sum(stationary_mass)
    end
    return mass ./ dx_stacked
end

"""
    StateEvolution(g, sm, times; dt=nothing, save=:all, callback=nothing,
                   stationary=false, workspace=nothing, normalize=true)

Evolve an initial density to strictly increasing real observation `times`.
The first time labels the initial state. `dt` limits the internal step size;
without it, take one implicit-Euler step per observation interval.

`save=:all` stores only observation-time densities, `:final` stores only the
last, and `:none` stores no trajectory. `final_density` is always available.
`callback(t, cell_masses)` runs at every observation time, including the first,
independently of `save`. Its mass buffer is borrowed: do not mutate or retain it.
This permits streaming moments without storing a distribution history.

Stationary computation is opt-in via `stationary=true`, or can be requested
later with `StationaryDistribution(result)`. No input array is mutated.

The legacy integer argument `T` means T observations at `0:T-1`;
`implicit_steps` specifies the number of steps per unit interval.
"""
function StateEvolution(g::AbstractVector, A_t::SparseMatrixCSC, times::AbstractVector,
                        v_dim, dx_stacked; dt = nothing, save = :all,
                        callback = nothing, stationary = false, workspace = nothing,
                        normalize = true)
    observations = Float64.(times)
    !isempty(observations) && all(isfinite, observations) && all(diff(observations) .> 0) ||
        throw(ArgumentError("Observation times must be finite, nonempty, and strictly increasing"))
    save in (:all, :final, :none) || throw(ArgumentError("save must be :all, :final, or :none"))
    dt === nothing || (isfinite(dt) && dt > 0) || throw(ArgumentError("dt must be finite and positive"))
    length(g) == prod(v_dim) == size(A_t, 1) || throw(DimensionMismatch("Density, grid shape, and generator must agree"))
    all(x -> isfinite(x) && x >= 0, g) && weighted_mass(g, dx_stacked) > 0 ||
        throw(ArgumentError("Initial density must be finite, nonnegative, and have positive mass"))
    first_step = length(observations) > 1 ? observations[2] - observations[1] : 1.0
    first_step = dt === nothing ? first_step : min(dt, first_step)
    blocksize = length(v_dim) > 2 ? prod(v_dim[1:2]) : length(g)
    ws = if workspace === nothing
        EvolutionWorkspace(A_t, dx_stacked; time_step = first_step, blocksize = blocksize)
    else
        update_evolution!(workspace, A_t; time_step = first_step, dx_stacked = dx_stacked)
    end
    T = eltype(ws.mass)
    initial = Vector{T}(g)
    normalize && normalize_by_weighted_mass!(initial, ws.dx_stacked)
    ws.mass .= ws.dx_stacked .* initial
    saved_times = save == :all ? observations : save == :final ? observations[end:end] : Float64[]
    saved = Matrix{T}(undef, length(g), length(saved_times))
    saved_index = 0
    for i in eachindex(observations)
        if i > 1
            interval = observations[i] - observations[i-1]
            # Equal substeps land exactly on the requested time without a
            # floating-point remainder creating a spurious extra tiny step.
            ratio = dt === nothing ? 1.0 : interval / dt
            steps = max(1, ceil(Int, ratio - 8eps(ratio)))
            step = interval / steps
            step == ws.time_step || update_evolution!(ws; time_step = step)
            for _ in 1:steps
                step_mass!(ws)
            end
        end
        if save == :all || (save == :final && i == length(observations))
            saved_index += 1
            @views saved[:, saved_index] .= ws.mass ./ ws.dx_stacked
        end
        callback === nothing || callback(observations[i], ws.mass)
    end
    final = ws.mass ./ ws.dx_stacked
    masses = reshape(saved .* ws.dx_stacked, (v_dim..., length(saved_times)))
    stationary_density = stationary ? StationaryDistribution(ws.A_t, ws.dx_stacked;
        initial_density = initial) : nothing
    return StateEvolution(saved, saved_times, initial, copy(ws.A_t), copy(ws.dx_stacked),
        masses, stationary_density, final, observations[end])
end

function StateEvolution(g::AbstractVector, A_t::SparseMatrixCSC, T::Int, v_dim, dx_stacked;
                        implicit_steps = 1, kwargs...)
    T >= 1 && implicit_steps isa Integer && implicit_steps >= 1 ||
        throw(ArgumentError("T and implicit_steps must be positive integers"))
    return StateEvolution(g, A_t, collect(0:T-1), v_dim, dx_stacked;
        dt = 1 / implicit_steps, kwargs...)
end

function evolution_grid(sm::SolvedModel)
    dims = size(sm.value.v)
    k = collect(view(reshape(sm.variables.k, dims[1], :), :, 1))
    return dims, create_dx_stacked(k, prod(dims) ÷ length(k))
end

function StateEvolution(g, sm::SolvedModel, times; kwargs...)
    dims, weights = evolution_grid(sm)
    return StateEvolution(g, sparse(sm.value.A'), times, dims, weights; kwargs...)
end

function StationaryDistribution(evolution::StateEvolution)
    return StationaryDistribution(evolution.A, evolution.dx_stacked; initial_density = evolution.g)
end

function StationaryDistribution(sm::SolvedModel; initial_density = nothing)
    _, weights = evolution_grid(sm)
    return StationaryDistribution(sparse(sm.value.A'), weights; initial_density = initial_density)
end

# helper to find group ids for plotting when the value function has been stacked
function create_group_ids(sm::SolvedModel)
    v_dim = size(sm.value.v)
    if length(v_dim) == 1
        col_ids = fill(1, v_dim[1])
        row_ids = collect(1:v_dim[1])
    else 
        col_ids = repeat(1:v_dim[2], inner = v_dim[1])
        row_ids = repeat(1:v_dim[1], outer = v_dim[2])
    end
    return col_ids, row_ids
end