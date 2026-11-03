
struct StateEvolution{T <: Real}
    S::Array{T}
    times::Vector{Int}
    g::Vector{T}
    A::SparseMatrixCSC{T, Int}
    dx_stacked::Vector{T} # grid spacing for mass conservation
    cell_masses::Array{T}
    stationary_dist::Array{T}
end

# Method to extract a column based on a time index using binary search
function Base.getindex(s::StateEvolution, t)
    time_index = Base.searchsortedfirst.(Ref(s.times), t)
    if time_index > length(s.times) || s.times[time_index] != t
        error("Time $t not found in times vector")
    end
    return s.E_S[:, time_index]
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
function iterate_g(g, A_t, dx_stacked; time_step = 1)
    evolution = I - time_step * A_t
    return (evolution \ (dx_stacked .* g)) ./ dx_stacked
end

function iterate_g!(g, A_t, dx_stacked; time_step = 1)
    g .= iterate_g(g, A_t, dx_stacked; time_step = time_step)
    return g
end

function StateEvolution(S::Array{T}, times::Vector{T}, g::Vector{T}, A::SparseMatrixCSC) where T <: Real
    return StateEvolution{T}(S, times, g, A)
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

# Given a stopping time T, iterate the state evolution T times with a timestep of 1/implicit_steps
function StateEvolution(g::Vector{R}, A_t::SparseMatrixCSC, T::Int, v_dim, dx_stacked; implicit_steps = 1) where {R <: Real}
    T_total = T*implicit_steps
    S = zeros((size(g, 1), T_total))
    # this will do nothing if g is already normalized - but best to makes sure
    g_norm = normalize_by_weighted_mass(g, dx_stacked)
    S[:, 1] .= g_norm
    for t in 2:T_total
        S[:, t] .= iterate_g(S[:, t-1], A_t, dx_stacked; time_step = 1.0/implicit_steps)
    end
    
    S_thinned = S[:, 1:implicit_steps:end]
    T_thinned = size(S_thinned, 2)
    T_thinned_vec = collect(0:(T_thinned - 1)) 

    cell_mass_all =  S_thinned .* dx_stacked
    full_shape = (v_dim..., T_thinned)
    cell_mass_array = reshape(cell_mass_all, full_shape)

    # Select the stationary mixture implied by the initial distribution.
    stationary_dist = StationaryDistribution(A_t, dx_stacked; initial_density = g_norm)
    return StateEvolution(S_thinned, T_thinned_vec, g, A_t, dx_stacked, cell_mass_array, stationary_dist)
end

# Given a vector of times, iterate the state evolution to each time step
# using the implicit formula to jump straight to that time step
function StateEvolution(g::Vector{T}, A_t::SparseMatrixCSC, times::Vector, v_dim, dx_stacked) where T <: Real
    S = zeros((size(g, 1), length(times)))
    # this will do nothing if g is already normalized - but best to makes sure
    g_norm = normalize_by_weighted_mass(g, dx_stacked)
    S[:, 1] .= g_norm
    i = 1
    for t in times[2:end]
        i += 1
        S[:, i] .= iterate_g(S[:, i-1], A_t, dx_stacked, time_step = t)
    end
    cell_mass_all = S .* dx_stacked 
    full_shape = (v_dim..., length(times))
    cell_mass_array = reshape(cell_mass_all, full_shape)

    # Select the stationary mixture implied by the initial distribution.
    stationary_dist = StationaryDistribution(A_t, dx_stacked; initial_density = g_norm)

    return StateEvolution(S, times, g, A_t, dx_stacked, cell_mass_array, stationary_dist)
end


function StateEvolution(g, sm::SolvedModel{S}, T::Int; implicit_steps = 1) where {S <: Model}
    # get all ks along k - using first dim of other states
    slice_tuple = CartesianIndices((1:size(sm.variables.k, 1), fill(1, ndims(sm.variables.k) - 1)...))
    if S <: StochasticModel
        Nstate = prod(size(sm.variables.z)[2:end])
    else
        Nstate = 1
    end
    k = sm.variables.k[slice_tuple][:]
    dx_stacked = create_dx_stacked(k, Nstate)
    A_t = sparse(sm.value.A')
    return StateEvolution(g, A_t, T, size(sm.value.v), dx_stacked, implicit_steps = implicit_steps)
end

function StateEvolution(g, sm::SolvedModel{S}, times::Vector) where {S <: Model}
    slice_tuple = CartesianIndices((1:size(sm.variables.k, 1), fill(1, ndims(sm.variables.k) - 1)...))
    k = sm.variables.k[slice_tuple][:]
    Nstate = prod(size(sm.value.v)) ÷ length(k)
    dx_stacked = create_dx_stacked(k, Nstate)
    A_t = sparse(sm.value.A')
    return StateEvolution(g, A_t, times, size(sm.value.v), dx_stacked)
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