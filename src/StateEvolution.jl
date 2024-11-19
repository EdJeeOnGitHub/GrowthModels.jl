
struct StateEvolution{T <: Real}
    S::Array{T}
    times::Vector{Int}
    g::Vector{T}
    A::SparseMatrixCSC{T, Int}
    E_S::Array{T}
    E_stationary::Array{T}
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

function iterate_g(g, A_t, grid_diag; time_step = 1)
    g_tilde = grid_diag * g
    g_new_tilde = iterate_g(g_tilde, A_t, time_step = time_step)
    g_new = grid_diag \ g_new_tilde
    return g_new
end

function iterate_g!(g, A_t, grid_diag; time_step = 1)
    g_tilde = grid_diag * g
    g_new_tilde = iterate_g(g_tilde, A_t, time_step = time_step)
    g .= grid_diag \ g_new_tilde
end

function StateEvolution(S::Array{T}, times::Vector{T}, g::Vector{T}, A::SparseMatrixCSC) where T <: Real
    return StateEvolution{T}(S, times, g, A)
end


function create_grid_diag(x, N_second_dim)
    dxf, dxb = generate_dx(x)
    dx_tilde = 0.5 * (dxf + dxb)
    dx_tilde[1] = 0.5*dxf[1]
    dx_tilde[end] = 0.5*dxb[end]
    dx_tilde_stacked = repeat(dx_tilde, N_second_dim)
    grid_diag = spdiagm(0 => dx_tilde_stacked)
    return grid_diag
end

function StationaryDistribution(A_t::SparseMatrixCSC, grid_diag) 
   # A matrix from VFI
   AT = copy(A_t)
   dim_b = size(A_t, 1)
   # We invert this with "infinite" time step to get stationary distribution
   # However, need to do two things:
   # 1. Add a row to AT to ensure it is invertible
   # 2. Rescale the problem since grid may be uneven - this is where grid_diag 
   #   comes in
   b = zeros(dim_b) 
   i_fix = dim_b ÷ 2
   b[i_fix] = 0.1
   row = zeros(dim_b)
   row[i_fix] = 1.0
   AT[i_fix, :] = row
   gg_tilde = AT \ b
   g_sum = sum(gg_tilde)
   gg_tilde = gg_tilde ./ g_sum
   gg = grid_diag \ gg_tilde
   return gg
end

# Given a stopping time T, iterate the state evolution T times with a timestep of 
# 1 unit each time
function StateEvolution(g::Vector{R}, A_t::SparseMatrixCSC, T::Int, v_dim, grid_diag) where {R <: Real}
    S = zeros((size(g, 1), T))
    S[:, 1] .= g
    for t in 2:T
        S[:, t] .= iterate_g(S[:, t-1], A_t, grid_diag; time_step = 1)
    end
    

    # if no shock dimension, return S
    if v_dim[1] == size(g, 1)
        E_S = S
        # also don't use infinite time step - just pass last time step
        E_stationary = E_S[:, end]
    else 
        # otherwise, average
        E_S = sum(reshape(S, (v_dim..., T)), dims = 2) |> x -> dropdims(x, dims = 2)
        # use infinite time step to get stationary distribution
        stationary_distribution = StationaryDistribution(A_t, grid_diag)
        E_stationary = sum(reshape(stationary_distribution, v_dim), dims = 2) |> x -> dropdims(x, dims = 2)
    end
    return StateEvolution(S, collect(0:(T-1)), g, A_t, E_S, E_stationary)
end

# Given a vector of times, iterate the state evolution to each time step
# using the implicit formula to jump straight to that time step
function StateEvolution(g::Vector{T}, A_t::SparseMatrixCSC, times::Vector, v_dim, grid_diag) where T <: Real
    S = zeros((size(g, 1), length(times)))
    S[:, 1] .= g
    i = 1
    for t in times[2:end]
        i += 1
        S[:, i] .= iterate_g(S[:, i-1], A_t, grid_diag, time_step = t)
    end
    # if no shock dimension return S
    if v_dim[1] == size(g, 1)
        E_S = S
        # also don't use infinite time step - just pass last time step
        E_stationary = E_S[:, end]
    else
        # otherwise, average
        E_S = sum(reshape(S, (v_dim..., T)), dims = 2) |> x -> dropdims(x, dims = 2)
        # use infinite time step to get stationary distribution
        stationary_distribution = StationaryDistribution(A_t, grid_diag)
        E_stationary = sum(reshape(stationary_distribution, v_dim), dims = 2) |> x -> dropdims(x, dims = 2)
    end
    return StateEvolution(S, times, g, A_t, E_S, E_stationary)
end


function StateEvolution(g, sm::SolvedModel{S}, T::Int) where {S <: Model}
    # get all ks along k - using first dim of other states
    slice_tuple = CartesianIndices((1:size(sm.variables.k, 1), fill(1, ndims(sm.variables.k) - 1)...))
    if S <: StochasticModel
        Nstate = prod(size(sm.variables.z)[2:end])
    else
        Nstate = 1
    end
    k = sm.variables.k[slice_tuple][:]
    grid_diag = create_grid_diag(k, Nstate)
    A_t = sparse(sm.value.A')
    return StateEvolution(g, A_t, T, size(sm.value.v), grid_diag)
end

function StateEvolution(g, sm::SolvedModel{S}, times::Vector) where {S <: Model}
    if S <: StochasticModel 
        dz = size(sm.variables.z, 2)
        k = sm.variables.k[:, 1]
    else
        dz = 1
        k = sm.variables.k
    end
    grid_diag = create_grid_diag(k, dz)
    A_t = sparse(sm.value.A')
    return StateEvolution(g, A_t, times, size(sm.value.v), grid_diag)
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