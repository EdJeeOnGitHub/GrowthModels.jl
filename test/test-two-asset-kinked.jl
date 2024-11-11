using GrowthModels
using Test
using CSV, Tables

using LinearAlgebra, SparseArrays



m = StochasticTwoAssetKinkedModel(nothing) 
hps = StateSpaceHyperParams(m)
state = StateSpace(m, hps)
init_value = Value(state)

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

function two_asset_kinked_FOC(p_a, p_b, a, χ_0, χ_1)
    d = min.(p_a ./ p_b .- 1 .+ χ_0, 0) .* a ./ χ_1 .+ max.(p_a ./ p_b .- 1 .- χ_0, 0) .* a ./ χ_1
    return d
end


# function update_v(m::StochasticModel{T, S}, value::Value{T, N_v}, state::StateSpace, hyperparams::StateSpaceHyperParams, diffusion_matrix; iter = 0, crit = 10^(-6), Delta = 1000, verbose = true) where {T, N_v, S <: StochasticProcess}
    # (; γ, α, ρ, δ) = m
    (; ρ, r_a, r_b_neg, r_b_pos, w, ξ, γ, χ_1, χ_0) = m
    (; v, dVf, dVb, dV0, dist) = value
    b, a, z = state[:b], state[:a], state[:z]' # y isn't really a state but avoid computing it each iteration this way
    y_b  = state.aux_state[:y_b]
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

    tau = 10
    Rb = permutedims(repeat(y_b, 1, 1, 50 ), (1, 3, 2))
    raa = r_a .* (1 .- (1.33 .* amax ./ a) .^ (1 - tau))
    Ra = repeat(reshape(raa, 1, :), Nb, 1, 2)

    
    Bswitch = diffusion_matrix    

    VbF = dVf[1, :, :, :] 
    VbB = dVb[1, :, :, :]

    VaF = dVf[2, :, :, :]
    VaB = dVb[2, :, :, :]

    v = (((1-ξ) * w* zzz + r_a .* aaa + r_b_neg .* bbb).^(1-γ))/(1-γ)/ρ;
    V = v
    Nb
    # deriv wrt b
    # Forward difference
    VbF[1:Nb-1, :, :] .= (V[2:Nb, :, :] - V[1:Nb-1, :, :]) ./ db
    VbF[Nb, :, :] .= ((1 - ξ) .* w .* zzz[Nb, :, :] + Rb[Nb, :, :] .* bmax) .^ (-γ) # State constraint at bmax
    # backward difference
    VbB[2:Nb, :, :] .= (V[2:Nb, :, :] - V[1:Nb-1, :, :]) ./ db
    VbB[1, :, :] .= ((1 - ξ) .* w .* zzz[1, :, :] + Rb[1, :, :] .* bmin) .^ (-γ) # State constraint at bmin

    # deriv wrt a
    VaF[:, 1:Na-1, :] .= (V[:, 2:Na, :] - V[:, 1:Na-1, :]) ./ da
    VaB[:, 2:Na, :] .= (V[:, 2:Na, :] - V[:, 1:Na-1, :]) ./ da

    VaB
    VbB

    aaa

    # consumption
    c_B = max.(VbB, eps()).^(-1 / γ)
    c_F = max.(VbF, eps()).^(-1 / γ)

    dBB = two_asset_kinked_FOC(VaB, VbB, aaa, χ_0, χ_1)
    dFB = two_asset_kinked_FOC(VaB, VbF, aaa, χ_0, χ_1)
    dBF = two_asset_kinked_FOC(VaF, VbB, aaa, χ_0, χ_1)
    dFF = two_asset_kinked_FOC(VaF, VbF, aaa, χ_0, χ_1)
    # Upwind stuff
    # Backwards
    d_B = (dBF .> 0) .* dBF .+ (dBB .< 0) .* dBB
    # state constraints at amin and amax
    d_B[:, 1, :] = (dBF[:, 1, :] .> eps()) .* dBF[:, 1, :] # at d >= 0 at amin, don't use VaB[:, 1, :]
    d_B[:, end, :] = (dBB[:, end, :] .< -eps()) .* dBB[:, end, :] # at d <= 0 at amax, don't use VaF[:, end, :]
    d_B[1, 1, :] = max.(d_B[1, 1, :], 0)
    # splitting drift of b and upwind separately
    sc_B = (1 - ξ) .* w .* zzz .+ Rb .* bbb - c_B
    sd_B = (-d_B - GrowthModels.StochasticTwoAssetKinkedModel_cost_adjustment(d_B, aaa, χ_0, χ_1))
    


    # Forwards
    d_F = (dFF .> 0) .* dFF .+ (dFB .< 0) .* dFB
    # state constraints at amin/amax
    d_F[:, 1, :] = (dFF[:, 1, :] .> eps()) .* dFF[:, 1, :] # at d >= 0 at amin, don't use VaB[:, 1, :]
    d_F[:, end, :] = (dFB[:, end, :] .< -eps()) .* dFB[:, end, :] # at d <= 0 at amax, don't use VaF[:, end, :]
    # splitting drift
    sc_F = ( 1 - ξ) .* w .* zzz .+ Rb .* bbb - c_F
    sd_F = (-d_F - GrowthModels.StochasticTwoAssetKinkedModel_cost_adjustment(d_F, aaa, χ_0, χ_1))
    sd_F[end, :, :] = min.(sd_F[end, :, :], 0) # at d <= 0 at amax

    Ic_B = (sc_B .< -eps())
    Ic_F = (sc_F .> eps()) .* (1 .- Ic_B)
    Ic_0 = 1 .- Ic_F .- Ic_B

    Id_F = (sd_F .> eps())
    Id_B = (sd_B .< -eps()) .* (1 .- Id_F)
    Id_B[1, :, :] .= 0
    Id_F[end, :, :] .= 0
    Id_B[end, :, :] .= 1 # don't use VbF at bmax so don't pick up artificial state constraint
    Id_0 = 1 .- Id_F .- Id_B

    c_0 = (1 - ξ) .* w .* zzz .+ Rb .* bbb
    c = c_B .* Ic_B .+ c_F .* Ic_F .+ c_0 .* Ic_0
    u = c.^(1 - γ) / (1 - γ)

    # matrix for evolution
    X = (-Ic_B .* sc_B  - Id_B .* sd_B) ./ db
    Y = (Ic_B .* sc_B - Ic_F .* sc_F) ./ db + (Id_B .* sd_B - Id_F .* sd_F) ./ db
    Z = (Ic_F .* sc_F + Id_F .* sd_F) ./ db


    # total_length = Nz*Nk
    # udiag = Vector{eltype(Z)}(undef, total_length - 1)
    # cdiag = reshape(Y, Nz*Nk)  # Assuming Y is already a matrix or array that matches the dimensions
    # ldiag = Vector{eltype(X)}(undef, total_length - 1)
    ## start here
    total_length = Nb*Na
    cdiag = reshape(Y, total_length, 2)
    udiag = Array{eltype(Z)}(undef, (total_length, 2))
    ldiag = Array{eltype(X)}(undef, (total_length, 2))

    ldiag[1:Nb-1, :] .= X[2:Nb, 1, :]
    udiag[2:Nb, :] .= Z[1:Nb-1, 1, :] 

    # I = Nb
    # J = Na

    for j in 2:Na
        # @show j
        # j = 50
        ldiag[1:j*Nb, :] .= [ldiag[1:(j-1) * Nb, :]; X[2:Nb, j, :]; zeros(1, Nz)]
        udiag[1:j*Nb, :] .= [udiag[1:(j - 1) * Nb, :]; zeros(1, Nz); Z[1:Nb-1, j, :]]
    end

    B_1 = spdiagm(0 => cdiag[:, 1], 1 => udiag[1:end-1, 1], -1 => ldiag[2:end, 1])
    B_2 = spdiagm(0 => cdiag[:, 2], 1 => udiag[1:end-1, 2], -1 => ldiag[2:end, 2])
    BB = [B_1 spzeros(total_length, total_length);
          spzeros(total_length, total_length) B_2]



    # quantities
    c_B = max.(VbB, eps()).^(-1 / γ)
    c_F = max.(VbF, eps()).^(-1 / γ)
    dBB = 


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

