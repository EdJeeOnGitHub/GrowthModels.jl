
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

function iterate_g(g, A_t, dx_stacked; time_step = 1)
    evolution = sparse(I, size(A_t)) - time_step * A_t
    new_g = evolution \ g
    # normalize with grid spacing
    normalized_g = normalize_by_weighted_mass(new_g, dx_stacked)
    return normalized_g
end

function iterate_g!(g, A_t, dx_stacked; time_step = 1)
    evolution = sparse(I, size(A_t)) - time_step * A_t
    g .= evolution \ g
    # normalize with grid spacing
    normalize_by_weighted_mass!(g, dx_stacked)
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
    # dx_stacked = spdiagm(0 => dx_tilde_stacked)
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

function StationaryDistribution(A_t::SparseMatrixCSC, dx_stacked) 
   dx_stacked_mat = spdiagm(0 => dx_stacked)
   # A matrix from VFI
   AT = copy(A_t)
   dim_b = size(A_t, 1)
   # We invert this with "infinite" time step to get stationary distribution
   # However, need to do two things:
   # 1. Add a row to AT to ensure it is invertible
   # 2. Rescale the problem since grid may be uneven - this is where dx_stacked 
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
   gg = dx_stacked_mat \ gg_tilde
   return gg
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

    # if no shock dimension, return S
    stationary_dist = v_dim[1] == size(g, 1) ? S[:, end] : StationaryDistribution(A_t, dx_stacked)
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

    # if no shock dimension, return S
    stationary_dist = v_dim[1] == size(g, 1) ? S[:, end] : StationaryDistribution(A_t, dx_stacked)

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
    if S <: StochasticModel 
        dz = size(sm.variables.z, 2)
        k = sm.variables.k[:, 1]
    else
        dz = 1
        k = sm.variables.k
    end
    dx_stacked = create_dx_stacked(k, dz)
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