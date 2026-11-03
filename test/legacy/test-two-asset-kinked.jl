using GrowthModels
using Test
using CSV, Tables

using LinearAlgebra, SparseArrays



m = StochasticTwoAssetKinkedModel(nothing) 
hps = StateSpaceHyperParams(m)
state = StateSpace(m, hps)
init_value = Value(state)


init_value.v

v_dim = size(init_value.v)

# fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000)

stochasticprocess = m.stochasticprocess
m = m
value = init_value
state = state
hyperparams = hps

# diffusion_matrix = 
iter = 0 
crit = 10^(-6)
Delta = 1000
verbose = true

diffusion_matrix = GrowthModels.construct_diffusion_matrix(m.stochasticprocess, state, hps)



# function update_v(m::StochasticModel{T, S}, value::Value{T, N_v}, state::StateSpace, hyperparams::StateSpaceHyperParams, diffusion_matrix; iter = 0, crit = 10^(-6), Delta = 1000, verbose = true) where {T, N_v, S <: StochasticProcess}
    # (; γ, α, ρ, δ) = m
    (; v, dVf, dVb, dV0, dist) = value
    b, a, z = state[:b], state[:a], state[:z]' # y isn't really a state but avoid computing it each iteration this way
    y_b, y_a = state.aux_state[:y_b], state.aux_state[:y_a]
    b_hps = hyperparams[:b]
    a_hps = hyperparams[:a]
    z_hps = hyperparams[:z]


    Nb, db, bmax, bmin = b_hps.N, b_hps.dx, b_hps.xmax, b_hps.xmin
    Na, da, amax, amin = a_hps.N, a_hps.dx, a_hps.xmax, a_hps.xmin
    Nz, dz, zmax, zmin = z_hps.N, z_hps.dx, z_hps.xmax, z_hps.xmin


    bb = repeat(reshape(b, :, 1), 1, Na)
    aa = repeat(reshape(a, 1, :), Nb, 1)
    zz = repeat(reshape(z, 1, :), Na, 1)

    bbb =  repeat(bb, 1, 1, 2) 
    aaa = repeat(aa, 1, 1, 2)
    zzz = permutedims(repeat(zz, 1, 1, 100), (3, 1, 2))


    Bswitch = diffusion_matrix    


    

    dVf 

    V = v

    # Forward difference
    dVf[1:Nk-1, :] .= (V[2:Nk, :] - V[1:Nk-1, :]) ./ dk
    dVf[Nk, :] .= (y[Nk, :] .- δ .* k[Nk, :]) .^ (-γ) # State constraint at kmax

    # Backward difference
    dVb[2:Nk, :] .= (V[2:Nk, :] - V[1:Nk-1, :]) ./ dk
    dVb[1, :] .= (y[1, :] .- δ .* k[1, :]).^(-γ) # State constraint at kmin

    # Indicator whether value function is concave
    I_concave = dVb .> dVf

    # Consumption and savings with forward difference
    cf = max.(dVf, eps()).^(-1 / γ)
    sf = y - δ .* kk - cf
    # Consumption and savings with backward difference
    cb = max.(dVb, eps()).^(-1 / γ)
    sb = y - δ .* kk - cb
    # Consumption and derivative of value function at steady state
    c0 = y - δ .* kk
    dV0 .= max.(c0, eps()).^(-γ)

    # Decision on forward or backward differences based on the sign of the drift
    If = sf .> 0 # positive drift -> forward difference
    Ib = sb .< 0 # negative drift -> backward difference
    I0 = 1 .- If .- Ib # at steady state

    V_Upwind = dVf .* If + dVb .* Ib + dV0 .* I0

    c = max.(V_Upwind, eps()).^(-1 / γ)
    u = c.^(1 - γ) / (1 - γ)

    # Construct matrix A
    X = -min.(sb, 0) ./ dk
    Y = -max.(sf, 0) ./ dk + min.(sb, 0) ./ dk
    Z = max.(sf, 0) ./ dk



    ## start here
    total_length = Nz*Nk
    udiag = Vector{eltype(Z)}(undef, total_length - 1)
    cdiag = reshape(Y, Nz*Nk)  # Assuming Y is already a matrix or array that matches the dimensions
    ldiag = Vector{eltype(X)}(undef, total_length - 1)

    index = 1
    for j in 1:Nz
        if j != Nz
            segment = vcat(Z[1:Nk-1, j], 0.0)  # Include 0.0 for all but the last column
            len = Nk  # Nk-1 elements plus a 0.0
        else
            segment = Z[1:Nk-1, j]  # Do not include 0.0 for the last column
            len = Nk - 1  # Only Nk-1 elements
        end
        
        udiag[index:index+len-1] .= segment
        index += len  # Adjust index for the next iteration
    end
    # Fill the first part of lowdiag without prepending 0
    ldiag[1:Nk-1] = X[2:end, 1]
    # Index to keep track of the position in lowdiag
    index = Nk
    for j in 2:Nz
        # Prepend 0 before adding elements from the jth column
        ldiag[index] = 0.0
        index += 1  # Move index after the 0
        
        # Slice assignment for elements from the jth column
        ldiag[index:index+Nk-2] = X[2:end, j]
        index += Nk-1  # Update index for the next iteration
    end


    AA = spdiagm(0 => cdiag, 1 => udiag, -1 => ldiag)


    A = AA + Bswitch
    A_err = abs.(sum(A, dims = 2))        
    if maximum(A_err) > 10^(-6)
        throw(ValueFunctionError("Improper Transition Matrix: $(maximum(A_err)) > 10^(-6)"))
    end    

    B = (1 / Delta + ρ) * sparse(I, size(A)) .- A



    u_stacked = reshape(u, Nk*Nz)
    V_stacked = reshape(V, Nk*Nz)

    b = u_stacked + V_stacked / Delta

    V_stacked = B \ b

    V = reshape(V_stacked, Nk, Nz)

    Vchange = V - v

    # If using forward diff, want this just to be value part
    distance = ForwardDiff.value(maximum(abs.(Vchange)))
    push!(dist, distance)

    if distance < crit
        if verbose
            println("Value Function Converged, Iteration = ", iter)
        end
        push!(dist, distance)
        value = Value{T, N_v}(
            v = V, 
            dVf = dVf, 
            dVb = dVb, 
            dV0 = dV0, 
            A = A,
            dist = dist,
            convergence_status = true,
            iter = iter
            )
        variables = (
            y = y, 
            k = kk, 
            z = zz,
            c = c, 
            If = If, 
            Ib = Ib, 
            I0 = I0
            )
        return value, iter, variables
    end

    value = Value{T,N_v}(
        v = V, 
        dVf = dVf, 
        dVb = dVb, 
        dV0 = dV0, 
        A = A,
        dist = dist,
        convergence_status = false,
        iter = iter
        )

    return value, iter
end

