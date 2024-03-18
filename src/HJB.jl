
function update_v(m::DeterministicModel, value::Value{T, N_v}, state::StateSpace, hyperparams::StateSpaceHyperParams; iter = 0, crit = 10^(-6), Delta = 1000, verbose = true) where {T, N_v}
    γ, ρ, δ = m.γ, m.ρ, m.δ
    (; v, dVf, dVb, dV0, dist) = value
    k, y = state[:k], state.aux_state[:y] # y isn't really a state but avoid computing it each iteration this way
    (; N, dx, xmax, xmin) = hyperparams[:k]
    dk, kmax, kmin = dx, xmax, xmin


    V = v
    # forward difference
    dVf[1:(N-1), 1] = (V[2:N, 1] .- V[1:(N-1), 1])/dk
    dVf[N, 1] = (y[N] - δ*kmax)^(-γ) # state constraint, for stability
    # backward difference
    dVb[2:N, 1] = (V[2:N, 1] .- V[1:(N-1), 1])/dk
    dVb[1, 1] = (y[1] - δ*kmin)^(-γ) # state constraint, for stability

    # consumption and savings with forward difference
    cf = max.(dVf[:, 1],eps()).^(-1/γ)
    muf = y - δ.*k - cf
    Hf = (cf.^(1-γ))/(1-γ) + dVf[:, 1].*muf

    # consumption and savings with backward difference
    cb = max.(dVb[:, 1],eps()).^(-1/γ)
    mub = y - δ.*k - cb
    Hb = (cb.^(1-γ))/(1-γ) + dVb[:, 1].*mub

    # consumption and derivative of value function at steady state
    c0 = y - δ.*k
    dV0[1:N, 1] = max.(c0, eps()).^(-γ)
    H0 = (c0.^(1-γ))/(1-γ)

    # dV_upwind makes a choice of forward or backward differences based on
    # the sign of the drift    
    Ineither = (1 .- (muf .> 0)) .* (1 .- (mub .< 0))
    Iunique = (mub .< 0) .* (1 .- (muf .> 0)) + (1 .- (mub .< 0)) .* (muf .> 0)
    Iboth = (mub .< 0) .* (muf .> 0)
    Ib = Iunique .* (mub .< 0) + Iboth .* (Hb .>= Hf)
    If = Iunique .* (muf .> 0) + Iboth .* (Hf .>= Hb)
    I0 = Ineither

    # consumption
    c  = cf .* If + cb .* Ib + c0 .* I0
    u = (c.^(1-γ))/(1-γ)

    # CONSTRUCT MATRIX
    X = -Ib .* mub/dk
    Y = -If .* muf/dk + Ib .* mub/dk
    Z = If .* muf/dk


    A = spdiagm(
        0 => Y,
        -1 => X[2:N],
        1 => Z[1:(N-1)]
    );
    A_err = abs.(sum(A, dims = 2))        
    if maximum(A_err) > 10^(-6)
        throw(ValueFunctionError("Improper Transition Matrix: $(maximum(A_err)) > 10^(-6)"))
    end    


    B = (ρ + 1/Delta) * sparse(I, N, N) .- A
    b = u + V/Delta
    V = B\b # SOLVE SYSTEM OF EQUATIONS
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
            dist = dist,
            convergence_status = true,
            iter = iter
            )
        variables = (
            y = y, 
            k = k, 
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
        dist = dist,
        convergence_status = false,
        iter = iter
        )

    return value, iter
end


"""
    update_value_function!(init_value, res)

If the HJB solver has converged, update the value function inplace. Uses ForwardDiff.value 
to ensure only the value and not dual components are passed. 

update_v is internal function used for value function iteration. update_value_function! is exported to the user so we can hotstart the VFI across optimization draws.
"""
function update_value_function!(init_value, res)
    if res.value.convergence_status == true
            init_value.v[:] = ForwardDiff.value.(res.value.v)
            init_value.dVf[:] = ForwardDiff.value.(res.value.dVf)
            init_value.dVb[:] = ForwardDiff.value.(res.value.dVb)
            init_value.dV0[:] = ForwardDiff.value.(res.value.dV0)
    end
end


function construct_diffusion_matrix(stochasticprocess::StochasticProcess, state::StateSpace, hyperparams::StateSpaceHyperParams)
    (; θ, σ) = stochasticprocess
    z = state[:z]
    k_hps, z_hps = hyperparams[:k], hyperparams[:z]

    Nk, dk, kmax, kmin = k_hps.N, k_hps.dx, k_hps.xmax, k_hps.xmin
    Nz, dz, zmax, zmin = z_hps.N, z_hps.dx, z_hps.xmax, z_hps.xmin

    dz2 = dz^2


    σ_sq = σ^2
    # drift
    mu = (-θ*log.(z) .+ σ_sq/2).*z
    # variance - Ito's
    s2 = σ_sq.*z.^2;

    yy = -s2/dz2 - mu/dz
    chi = s2/(2*dz2)
    zeta = mu/dz + s2/(2*dz2)
    
    off_diag_length = (Nz-1)*Nk

    ldiag = Vector{typeof(chi[2])}(undef, off_diag_length)
    cdiag = Vector{typeof(yy[1] + chi[1])}(undef, Nz*Nk)
    udiag = Vector{eltype(zeta[1])}(undef, off_diag_length)

    for j in 2:Nz
        ldiag[(j-2)*Nk+1:(j-1)*Nk] .= chi[j]
    end
    for j in 1:(Nz-1)
        udiag[(j-1)*Nk+1:(j)*Nk] .= zeta[j]
    end
    cdiag[1:Nk] .= chi[1] + yy[1]
    for j in 2:(Nz -1)
        cdiag[(j-1)*Nk+1:(j)*Nk] .= yy[j]
    end
    cdiag[end-Nk+1:end] .= zeta[end] + yy[end]

    # Construct B_switch matrix with corrected diagonals
    Bswitch = spdiagm(-Nk => ldiag, 0 => cdiag, Nk => udiag)
    return Bswitch
end

function update_v(m::StochasticModel, value::Value{T, N_v}, state::StateSpace, hyperparams::StateSpaceHyperParams, diffusion_matrix; iter = 0, crit = 10^(-6), Delta = 1000, verbose = true) where {T, N_v}
    (; γ, α, ρ, δ) = m
    (; v, dVf, dVb, dV0, dist) = value
    k, z = state[:k], state[:z]' # y isn't really a state but avoid computing it each iteration this way
    y = state.aux_state[:y]
    k_hps = hyperparams[:k]
    z_hps = hyperparams[:z]


    
    Nk, dk, kmax, kmin = k_hps.N, k_hps.dx, k_hps.xmax, k_hps.xmin
    Nz, dz, zmax, zmin = z_hps.N, z_hps.dx, z_hps.xmax, z_hps.xmin


    kk = repeat(reshape(k, :, 1), 1, Nz);
    zz = repeat(reshape(z, 1, :), Nk, 1);

    Bswitch = diffusion_matrix    

    V = v

    # Forward difference
    dVf[1:Nk-1, :] .= (V[2:Nk, :] - V[1:Nk-1, :]) ./ dk

    # Backward difference
    dVb[2:Nk, :] .= (V[2:Nk, :] - V[1:Nk-1, :]) ./ dk
    dVb[1, :] .= (y[1, :] .- δ .* k[1, :]).^(-γ) # State constraint at kmin

    # Indicator whether value function is concave
    I_concave = dVb .> dVf

    # Consumption and savings with forward difference
    cf = dVf.^(-1 / γ)
    sf = y - δ .* kk - cf
    # Consumption and savings with backward difference
    cb = dVb.^(-1 / γ)
    sb = y - δ .* kk - cb
    # Consumption and derivative of value function at steady state
    c0 = y - δ .* kk
    dV0 .= c0.^(-γ)

    # Decision on forward or backward differences based on the sign of the drift
    If = sf .> 0 # positive drift -> forward difference
    Ib = sb .< 0 # negative drift -> backward difference
    I0 = 1 .- If .- Ib # at steady state

    V_Upwind = dVf .* If + dVb .* Ib + dV0 .* I0

    c = V_Upwind.^(-1 / γ)
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
        dist = dist,
        convergence_status = false,
        iter = iter
        )

    return value, iter
end

initial_guess(m::GrowthModels.Model{T}, state) where {T <: Real} = state.aux_state[:y] .^ (1 - m.γ) / (1 - m.γ) / m.ρ

function solve_HJB(m::Model, hyperparams::StateSpaceHyperParams, state::StateSpace; init_value = Value(hyperparams), maxit = 1000, verbose = true)
    curr_iter = 0
    val = deepcopy(init_value)
    val.v[:] = initial_guess(m, state)
    for n in 1:maxit
        curr_iter += 1
        output_value, curr_iter = update_v(m, val, state, hyperparams, iter = n, verbose = verbose)
        if output_value.convergence_status
            fit_value, _, fit_variables = update_v(m, val, state, hyperparams, iter = curr_iter, verbose = verbose)
            return (value = fit_value, variables = fit_variables, iter = curr_iter)
            break
        end
        val = output_value
    end
    return (value = val, variables = nothing, iter = curr_iter)
end

function solve_HJB(m::Model, hyperparams::StateSpaceHyperParams; init_value = nothing, maxit = 1000, verbose = true)
    state = StateSpace(m, hyperparams)
    if isnothing(init_value)
        init_value = Value(state)
    end
    return solve_HJB(m, hyperparams, state; init_value = init_value, maxit = maxit, verbose = verbose)
end


function solve_HJB(m::StochasticModel, hyperparams::StateSpaceHyperParams, state::StateSpace; init_value = Value(hyperparams), maxit = 1000, verbose = true)
    curr_iter = 0
    val = deepcopy(init_value)
    val.v[:] = initial_guess(m, state)
    Bswitch = construct_diffusion_matrix(m.stochasticprocess, state, hyperparams)
    for n in 1:maxit
        curr_iter += 1
        output_value, curr_iter = update_v(m, val, state, hyperparams, Bswitch, iter = n, verbose = verbose)
        if output_value.convergence_status
            fit_value, _, fit_variables = update_v(m, val, state, hyperparams, Bswitch, iter = curr_iter, verbose = verbose)
            return (value = fit_value, variables = fit_variables, iter = curr_iter)
            break
        end
        val = output_value
    end
    return (value = val, variables = nothing, iter = curr_iter)
end



dV_Upwind(::Model, value::Value, variables::NamedTuple) = value.dVf .* variables.If .+ value.dVb .* variables.Ib .+ value.dV0 .* variables.I0
V_err(m::Model) = (value::Value, variables::NamedTuple) -> variables.c .^ (1-m.γ) / (1-m.γ) .+ dV_Upwind(m, value, variables) .* statespace_k_dot(m)(variables) .- m.ρ .* value.v

