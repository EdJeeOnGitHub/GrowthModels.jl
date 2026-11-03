# A fixed-pattern block solve. The maps refer to stored entries, including zeros,
# so changes in the selected drift do not invalidate the symbolic factorization.
struct SparseBlockSolver{T,F}
    blocks::Vector{SparseMatrixCSC{T,Int}}
    factors::Vector{F}
    ranges::Vector{UnitRange{Int}}
    maps::Vector{Vector{Int}}
    colptr::Vector{Int}
    rowval::Vector{Int}
end

_sparse_factor(A) = factor_sparse(A)

function SparseBlockSolver(A::SparseMatrixCSC; blocksize = size(A, 1))
    n = size(A, 1)
    size(A, 2) == n && n > 0 || throw(ArgumentError("Expected a nonempty square matrix"))
    blocksize isa Integer && blocksize > 0 && n % blocksize == 0 || throw(ArgumentError("Block size must divide the number of states"))
    ranges = [i:min(i + blocksize - 1, n) for i in 1:blocksize:n]
    maps = [Int[] for _ in ranges]
    for j in 1:n, ptr in nzrange(A, j)
        block = (j - 1) ÷ blocksize + 1
        if rowvals(A)[ptr] in ranges[block]
            push!(maps[block], ptr)
        elseif !iszero(nonzeros(A)[ptr])
            throw(ArgumentError("Generator couples states in different requested blocks"))
        end
    end
    blocks = [A[r, r] for r in ranges]
    factors = map(_sparse_factor, blocks)
    return SparseBlockSolver(blocks, factors, ranges, maps, copy(A.colptr), copy(rowvals(A)))
end

function refactor!(solver::SparseBlockSolver, A::SparseMatrixCSC)
    A.colptr == solver.colptr && rowvals(A) == solver.rowval ||
        throw(ArgumentError("Cannot reuse symbolic factorization after a sparsity-pattern change"))
    blocksize = length(first(solver.ranges))
    for col in axes(A, 2), ptr in nzrange(A, col)
        if (rowvals(A)[ptr] - 1) ÷ blocksize != (col - 1) ÷ blocksize && !iszero(nonzeros(A)[ptr])
            throw(ArgumentError("Generator couples states in different requested blocks"))
        end
    end
    for i in eachindex(solver.blocks)
        block = solver.blocks[i]
        map = solver.maps[i]
        for j in eachindex(map)
            nonzeros(block)[j] = nonzeros(A)[map[j]]
        end
        factor = solver.factors[i]
        refactor_sparse!(factor, block)
    end
    return solver
end

_solve_block!(out, factor, block, rhs) = solve_sparse_factor!(out, factor, rhs)

function solve_linear!(out, solver::SparseBlockSolver, rhs)
    for i in eachindex(solver.ranges)
        r = solver.ranges[i]
        _solve_block!(view(out, r), solver.factors[i], solver.blocks[i], view(rhs, r))
    end
    return out
end

# Discounted deterministic HJB systems are strictly row-diagonally-dominant
# tridiagonal M-matrices. Thomas elimination therefore needs no pivoting and
# works with both floating point and ForwardDiff dual numbers.
struct TridiagonalHJBSolver{T}
    diagonal::Vector{T}
    upper::Vector{T}
    multipliers::Vector{T}
end

function TridiagonalHJBSolver(A::SparseMatrixCSC{T}) where T
    n = size(A, 1)
    solver = TridiagonalHJBSolver(zeros(T, n), zeros(T, n - 1), zeros(T, n - 1))
    return refactor!(solver, A)
end

function refactor!(solver::TridiagonalHJBSolver, A::SparseMatrixCSC)
    n = length(solver.diagonal)
    solver.diagonal[1] = A[1, 1]
    for i in 1:n-1
        solver.upper[i] = A[i, i+1]
        solver.multipliers[i] = A[i+1, i] / solver.diagonal[i]
        solver.diagonal[i+1] = A[i+1, i+1] - solver.multipliers[i] * solver.upper[i]
    end
    return solver
end

function solve_linear!(out, solver::TridiagonalHJBSolver, rhs)
    n = length(out)
    out[1] = rhs[1]
    for i in 2:n
        out[i] = rhs[i] - solver.multipliers[i-1] * out[i-1]
    end
    out[n] /= solver.diagonal[n]
    for i in n-1:-1:1
        out[i] = (out[i] - solver.upper[i] * out[i+1]) / solver.diagonal[i]
    end
    return out
end
