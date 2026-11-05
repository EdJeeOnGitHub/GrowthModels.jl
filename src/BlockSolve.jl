# Solves the implicit HJB iteration on the independent blocks of its operator.
#
# Blocks are read from the generator's stored sparsity pattern, never from model
# type or axis names. A model that links two states through the operator therefore
# merges their blocks automatically, and a fully coupled model is simply one block:
# there is one code path, not a fast path beside a general one.
#
# The generator cannot reveal coupling outside the operator, such as a payoff that
# reads another state's value. iterate_block! therefore only ever sees its own
# block's arrays, so adding that kind of coupling requires changing this interface.
#
# Model code (`resources`, process generators) runs once, on the calling task, when
# the joint workspace is built. Worker tasks only run package numerics, so custom
# callables never need to be thread safe.

const HISTORY_ROW = NamedTuple{(:iteration,:update,:residual),Tuple{Int,Float64,Float64}}

"""Finest partition of `1:n` into contiguous ranges that no stored entry of `A` links.
Stored zeros count as links, so a rate that may become nonzero keeps states together.
States that are separable but interleaved are merged, which is valid but slower."""
function operator_blocks(A::SparseMatrixCSC)
    n = size(A, 1)
    reach = collect(1:n)
    for col in 1:n, ptr in nzrange(A, col)
        lo, hi = minmax(rowvals(A)[ptr], col)
        reach[lo] = max(reach[lo], hi)
    end
    blocks = UnitRange{Int}[]
    start, furthest = 1, 0
    for i in 1:n
        furthest = max(furthest, reach[i])
        if furthest == i
            push!(blocks, start:i)
            start = i + 1
        end
    end
    return blocks
end

"""Apply `f` to each item, on worker tasks when threads are available.
Uses `@spawn`, which composes with an enclosing threaded loop, where `@threads
:static` would throw. Errors are rethrown unwrapped, in item order, so a failure is
the same exception and the same block at any thread count."""
function run_blocks(f, items)
    (Threads.nthreads() == 1 || length(items) == 1) && return map(f, items)
    guarded(item) = try
        (true, f(item))
    catch err
        (false, err)
    end
    results = map(fetch, [Threads.@spawn(guarded(item)) for item in items])
    for (succeeded, result) in results
        succeeded || throw(result)
    end
    return [last(r) for r in results]
end

"""Midpoint-first `(block, parent)` levels. Parents are fixed in advance and finish
before their children start, so warm starts cannot depend on task scheduling."""
function warm_start_levels(n::Integer)
    levels = Vector{Tuple{Int,Int}}[]
    spans = [(1, n, 0)]
    while !isempty(spans)
        level = Tuple{Int,Int}[]
        next = Tuple{Int,Int,Int}[]
        for (lo, hi, parent) in spans
            lo > hi && continue
            mid = (lo + hi) ÷ 2
            push!(level, (mid, parent))
            push!(next, (lo, mid - 1, mid), (mid + 1, hi, mid))
        end
        isempty(level) || push!(levels, level)
        spans = next
    end
    return levels
end

"""An independent workspace for joint states `r`, which share no arrays with `ws`."""
function block_workspace(ws::HJBWorkspace, r::UnitRange{Int})
    Nk = size(ws.y, 1)
    length(r) % Nk == 0 || throw(ArgumentError("Operator blocks must contain whole capital columns"))
    shaped(x) = reshape(x[r], Nk, :)
    offset = ws.A.colptr[first(r)] - 1
    entries = offset+1:ws.A.colptr[last(r)+1]-1
    local_pointer(p) = p == 0 ? 0 : p - offset
    A = SparseMatrixCSC(length(r), length(r), ws.A.colptr[first(r):last(r)+1] .- offset,
        rowvals(ws.A)[entries] .- (first(r) - 1), nonzeros(ws.A)[entries])
    return HJBWorkspace(NamedTuple(), shaped(ws.y), shaped(ws.c0), copy(ws.dkf), copy(ws.dkb),
        shaped(ws.dVf), shaped(ws.dVb), shaped(ws.dV0), shaped(ws.c),
        shaped(ws.If), shaped(ws.Ib), shaped(ws.I0), ws.rhs[r], shaped(ws.next_v),
        A, copy(A), ws.diffusion_values[entries], local_pointer.(ws.diagonal[r]),
        local_pointer.(ws.backward[r]), local_pointer.(ws.forward[r]))
end

# BitArrays pack neighbouring states into one word, so concurrent writes to different
# blocks would race. solve_blocks calls this serially, after each level has finished.
function store_block!(ws::HJBWorkspace, block::HJBWorkspace, r::UnitRange{Int})
    start, n = first(r), length(r)
    for (destination, source) in ((ws.c, block.c), (ws.dVf, block.dVf), (ws.dVb, block.dVb),
            (ws.If, block.If), (ws.Ib, block.Ib), (ws.I0, block.I0))
        copyto!(destination, start, source, 1, n)
    end
    copyto!(nonzeros(ws.A), ws.A.colptr[start], nonzeros(block.A), 1, nnz(block.A))
    return ws
end

# A single capital column is tridiagonal whatever kind of model produced it.
block_solver(ws::HJBWorkspace) = length(ws.rhs) == size(ws.y, 1) ?
    TridiagonalHJBSolver(ws.B) : SparseBlockSolver(ws.B)

# Prepares the policy for `v`. Returns a description of the problem, or nothing.
function start_block!(ws, m, v, Delta)
    try
        prepare_hjb_step!(ws, m, v, Delta)
    catch err
        err isa Union{ValueFunctionError,DomainError} || rethrow()
        return "No finite initial value/policy pair is available: $(sprint(showerror,err))"
    end
    isfinite(nonlinear_residual(ws, m, v)) || return "No finite initial value/policy residual is available"
    return nothing
end

function iterate_block!(ws, m, v, options)
    (; Delta, maxiters, value_tolerance, residual_tolerance, history) = options
    trace = HISTORY_ROW[]
    update = Inf
    residual = nonlinear_residual(ws, m, v)
    outcome(iterations, stop) = (workspace=ws, value=v, iterations=iterations, stop=stop,
        update=update, residual=residual, trace=trace)
    maxiters == 0 && return outcome(0, :iteration_limit)
    solver = block_solver(ws)
    for n in 1:maxiters
        try
            n > 1 && refactor!(solver, ws.B)
            solve_linear!(vec(ws.next_v), solver, ws.rhs)
            update = hjb_distance(ws, v)
            prepare_hjb_step!(ws, m, ws.next_v, Delta)
            candidate_residual = nonlinear_residual(ws, m, ws.next_v)
            isfinite(candidate_residual) || throw(ValueFunctionError("Nonfinite candidate residual"))
            copyto!(v, ws.next_v)
            residual = candidate_residual
        catch err
            err isa Union{ValueFunctionError,DomainError,SingularException} || rethrow()
            prepare_hjb_step!(ws, m, v, Delta)
            residual = nonlinear_residual(ws, m, v)
            return outcome(n, :nonfinite)
        end
        history && push!(trace, HISTORY_ROW((n, update, residual)))
        update < value_tolerance && residual < residual_tolerance && admissible_hjb_policy(ws) &&
            return outcome(n, :converged)
    end
    return outcome(maxiters, :iteration_limit)
end

# `warm` comes from a neighbouring block, so an unusable one falls back to the cold
# guess. A caller's own initial value is never replaced: its failure is reported.
function solve_block(joint, r, m, start, warm, options)
    ws = block_workspace(joint, r)
    v = similar(ws.next_v)
    v .= warm === nothing ? reshape(view(start, r), size(v)) : warm
    problem = start_block!(ws, m, v, options.Delta)
    if warm !== nothing && problem !== nothing
        v .= reshape(view(start, r), size(v))
        problem = start_block!(ws, m, v, options.Delta)
    end
    problem === nothing || return (problem=problem,)
    initial = copy(v)
    return merge(iterate_block!(ws, m, v, options), (problem=nothing, initial=initial))
end

function warm_guess(outcomes, parent, r)
    parent == 0 && return nothing
    source = outcomes[parent]
    source.stop == :converged && length(source.value) == length(r) || return nothing
    return guess_values(source.value)
end

# Each iteration reports the worst block. A finished block repeats its last row,
# which is the state it holds for the rest of the solve.
function combined_history(outcomes)
    traces = [o.trace for o in outcomes if !isempty(o.trace)]
    isempty(traces) && return HISTORY_ROW[]
    return map(1:maximum(length, traces)) do n
        rows = [t[min(n, length(t))] for t in traces]
        HISTORY_ROW((n, maximum(row.update for row in rows), maximum(row.residual for row in rows)))
    end
end

"""Solve every operator block from the joint starting array `start`, then return one
GrowthSolution. With `warm=true` blocks after the first start from a converged
neighbour; `fit.initial_value` records the starting values actually used, so solving
again from it reproduces the same iterates. Nothing depends on the thread count."""
function solve_blocks(m, g, ws, start, warm, settings, blocks=operator_blocks(ws.A))
    levels = warm ? warm_start_levels(length(blocks)) :
        [[(b, 0) for b in eachindex(blocks)]]
    outcomes = Vector{Any}(undef, length(blocks))
    for level in levels
        results = run_blocks(level) do (b, parent)
            guess = warm_guess(outcomes, parent, blocks[b])
            solve_block(ws, blocks[b], m, start, guess, settings)
        end
        for ((b, _), result) in zip(level, results)
            result.problem === nothing || throw(ArgumentError(result.problem))
            outcomes[b] = result
        end
    end
    v, initial = similar(start), similar(start)
    for (r, o) in zip(blocks, outcomes)
        store_block!(ws, o.workspace, r)
        copyto!(v, first(r), o.value, 1, length(r))
        copyto!(initial, first(r), o.initial, 1, length(r))
    end
    stops = [o.stop for o in outcomes]
    stop = all(==(:converged), stops) ? :converged :
        any(==(:nonfinite), stops) ? :nonfinite : :iteration_limit
    report = (iterations=[o.iterations for o in outcomes], stop_reason=stops)
    return solution_result(m, g, ws, v, initial, settings, maximum(report.iterations), stop,
        maximum(o.update for o in outcomes), maximum(o.residual for o in outcomes),
        combined_history(outcomes), report)
end
