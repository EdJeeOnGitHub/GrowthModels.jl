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
    stationary_mass(A_t; initial_mass=nothing)

Limiting cell probabilities for a forward generator. Explicit initial mass
selects class weights, including absorption from transient states. Without it,
a unique closed communicating class is required. Density conversion belongs to
the distribution boundary.
"""
function stationary_mass(A_t::SparseMatrixCSC; initial_mass=nothing)
    check_generator(sparse(transpose(A_t)))
    n=size(A_t,1)
    classes,transient=closed_communicating_classes(A_t)
    if initial_mass === nothing
        length(classes) == 1 || throw(ArgumentError("Multiple stationary classes: supply an initial distribution to determine their masses"))
    else
        length(initial_mass) == n || throw(DimensionMismatch("Initial mass must match the generator"))
        validate_probability(initial_mass)
    end

    # Integrated transient occupancy solves -A_TT * occupation = initial_mass_T.
    # Its outflow into each closed class is that class's eventual absorbed mass.
    occupation = if initial_mass !== nothing && !isempty(transient)
        solve_sparse_factor(factor_sparse(-A_t[transient, transient]), initial_mass[transient])
    else
        nothing
    end
    mass = zeros(promote_type(Float64, eltype(A_t), initial_mass === nothing ? Float64 : eltype(initial_mass)), n)
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
    return mass
end

# Ability is fixed: population defaults select block masses, not arbitrary
# mixtures among closed classes inside a type. Explicit initial mass bypasses
# this default entirely.
function population_stationary_mass(A_t::SparseMatrixCSC,g::StateGrid,pop::Population)
    blocksize=preferred_blocksize(g)
    independent_blocksize(A_t,blocksize) == blocksize || throw(ArgumentError("Population types must be independent blocks"))
    T=promote_type(eltype(A_t),eltype(pop.shares))
    result=zeros(T,length(g))
    for (j,share) in enumerate(pop.shares)
        iszero(primal(share)) && continue
        indices=(j-1)*blocksize+1:j*blocksize
        result[indices] .= share .* stationary_mass(A_t[indices,indices])
    end
    return result
end

function forward_generator(fit::GrowthSolution)
    A=sparse(transpose(getfield(fit,:generator)))
    check_generator(sparse(transpose(A)))
    return A
end
