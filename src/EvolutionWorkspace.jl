"""
    EvolutionWorkspace(A_t, dx_stacked; time_step=1, blocksize=size(A_t, 1))

Cache an implicit-Euler factorization and mass/density buffers. `A_t` is the
transpose of the HJB generator. The workspace owns copies of the generator and
cell weights; use `update_evolution!` to change them. Reuse is safe across initial
distributions, but each concurrent simulation needs its own workspace.

`blocksize` can split independent fixed-ability groups. Cross-block transitions
are rejected. Changing the number of states requires a new workspace.
"""
mutable struct EvolutionWorkspace{T,S}
    A_t::SparseMatrixCSC{T,Int}
    dx_stacked::Vector{T}
    time_step::T
    system::SparseMatrixCSC{T,Int}
    solver::S
    mass::Vector{T}
    next_mass::Vector{T}
    density::Vector{T}
    blocksize::Int
    factorizations::Int
end

function validate_evolution_inputs(A_t, weights, time_step)
    n = size(A_t, 1)
    size(A_t, 2) == n && n > 0 || throw(ArgumentError("Expected a nonempty square generator"))
    length(weights) == n || throw(DimensionMismatch("One cell weight is required per state"))
    all(w -> isfinite(w) && w > 0, weights) || throw(ArgumentError("Cell weights must be finite and positive"))
    isfinite(time_step) && time_step > 0 || throw(ArgumentError("Time step must be finite and positive"))
    check_generator(sparse(transpose(A_t)))
end

function EvolutionWorkspace(A_t, dx_stacked; time_step = 1, blocksize = size(A_t, 1))
    validate_evolution_inputs(A_t, dx_stacked, time_step)
    T = promote_type(Float64, eltype(A_t), eltype(dx_stacked), typeof(time_step))
    generator = SparseMatrixCSC{T,Int}(sparse(A_t))
    generator = copy(generator)
    weights = Vector{T}(dx_stacked)
    system = sparse(I - time_step * generator)
    solver = SparseBlockSolver(system; blocksize = blocksize)
    n = length(weights)
    return EvolutionWorkspace(generator, weights, T(time_step), system, solver,
        zeros(T, n), zeros(T, n), zeros(T, n), blocksize, length(solver.blocks))
end

"""
    update_evolution!(workspace, A_t; time_step=workspace.time_step,
                      dx_stacked=workspace.dx_stacked)

Refresh the cache if generator values or timestep changed. A changed sparsity
pattern rebuilds symbolic analysis; otherwise only numerical factors are updated.
Changing cell weights alone updates the conversion buffers' convention and does
not require a new factorization. Inputs are copied, never retained by reference.
"""
function update_evolution!(ws::EvolutionWorkspace, A_t = ws.A_t;
                           time_step = ws.time_step, dx_stacked = ws.dx_stacked)
    size(A_t) == size(ws.A_t) || throw(DimensionMismatch("Construct a new workspace for a different state count"))
    changed = time_step != ws.time_step || A_t != ws.A_t
    if changed || dx_stacked != ws.dx_stacked
        validate_evolution_inputs(A_t, dx_stacked, time_step)
    end
    if changed
        generator = convert(typeof(ws.A_t), sparse(A_t))
        system = sparse(I - time_step * generator)
        # Construct/refactor before publishing the new generator snapshot.
        if system.colptr == ws.system.colptr && rowvals(system) == rowvals(ws.system)
            refactor!(ws.solver, system)
        else
            ws.solver = SparseBlockSolver(system; blocksize = ws.blocksize)
        end
        ws.system = system
        ws.A_t = copy(generator)
        ws.time_step = time_step
        ws.factorizations += length(ws.solver.blocks)
    end
    copyto!(ws.dx_stacked, dx_stacked)
    return ws
end

function step_mass!(ws::EvolutionWorkspace)
    solve_linear!(ws.next_mass, ws.solver, ws.mass)
    ws.mass, ws.next_mass = ws.next_mass, ws.mass
    return ws.mass
end

function iterate_g!(g::AbstractVector, ws::EvolutionWorkspace)
    length(g) == length(ws.mass) || throw(DimensionMismatch("Density must match the workspace"))
    ws.mass .= ws.dx_stacked .* g
    step_mass!(ws)
    g .= ws.mass ./ ws.dx_stacked
    return g
end

function iterate_g(g::AbstractVector, ws::EvolutionWorkspace)
    result = Vector{eltype(ws.mass)}(g)
    return iterate_g!(result, ws)
end
