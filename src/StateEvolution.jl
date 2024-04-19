
struct StateEvolution{T <: Real}
    S::Array{T}
    times::Vector{Int}
    g::Vector{T}
    A::SparseMatrixCSC{T, Int}
end

# Method to extract a column based on a time index using binary search
function Base.getindex(s::StateEvolution, t::Int)
    time_index = Base.searchsortedfirst(s.times, t)
    if time_index > length(s.times) || s.times[time_index] != t
        error("Time $t not found in times vector")
    end
    return s.S[:, time_index]
end

# Implicit Euler method
function iterate_g!(g, A_t; time_step = 1)
    I_A = sparse(I, size(A_t))
    evolution = I_A - time_step * A_t
    g .= evolution \ g
end
# Implicit Euler method
function iterate_g(g, A_t; time_step = 1)
    I_A = sparse(I, size(A_t))
    evolution = I_A - time_step * A_t
    return evolution \ g
end

function StateEvolution(S::Array{T}, times::Vector{T}, g::Vector{T}, A::SparseMatrixCSC) where T <: Real
    return StateEvolution{T}(S, times, g, A)
end

# Given a stopping time T, iterate the state evolution T times with a timestep of 
# 1 unit each time
function StateEvolution(g::Vector{R}, A_t::SparseMatrixCSC, T::Int) where {R <: Real}
    S = zeros((size(g, 1), T))
    S[:, 1] .= g
    for t in 2:T
        S[:, t] .= iterate_g(S[:, t-1], A_t)
    end
    return StateEvolution(S, collect(1:T), g, A_t)
end

# Given a vector of times, iterate the state evolution to each time step
# using the implicit formula to jump straight to that time step
function StateEvolution(g::Vector{T}, A_t::SparseMatrixCSC, times::Vector) where T <: Real
    S = zeros((size(g, 1), length(times)))
    S[:, 1] .= g
    i = 1
    for t in times[2:end]
        i += 1
        S[:, i] .= iterate_g(S[:, i-1], A_t, time_step = t)
    end
    return StateEvolution(S, times, g, A_t)
end


# helper to find group ids for plotting when the value function has been stacked
function create_group_ids(sm::SolvedModel)
    v_dim = size(sm.value.v)
    if length(v_dim) == 1
        group_ids = fill(1, v_dim[1])
    else 
        group_ids = repeat(1:v_dim[2], inner = v_dim[1])
    end
    return group_ids
end