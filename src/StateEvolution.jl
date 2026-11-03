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
        solve_sparse_factor(factor_sparse(-A_t[transient, transient]), initial_mass[transient])
    else
        nothing
    end
    mass = zeros(promote_type(Float64, eltype(A_t), eltype(dx_stacked), initial_mass === nothing ? Float64 : eltype(initial_mass)), n)
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
        stationary_mass = solve_sparse_factor(factor_sparse(block), rhs)
        mass[states] .= weight .* stationary_mass ./ sum(stationary_mass)
    end
    return mass ./ dx_stacked
end
