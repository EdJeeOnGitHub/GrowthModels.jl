
function update_v(m::Union{SkibaModel,SmoothSkibaModel,RamseyCassKoopmansModel}, value::Value, state::StateSpace, hyperparams::HyperParams; iter = 0, crit = 10^(-6), Delta = 1000, silent = false)
    γ, ρ, δ = m.γ, m.ρ, m.δ
    (; v, dVf, dVb, dV0, dist) = value
    (; k, y) = state # y isn't really a state but avoid computing it each iteration this way
    (; N, dk, kmax, kmin) = hyperparams


    V = v
    # forward difference
    dVf[1:(N-1)] = (V[2:N] .- V[1:(N-1)])/dk
    dVf[N] = (y[N] - δ*kmax)^(-γ) # state constraint, for stability
    # backward difference
    dVb[2:N] = (V[2:N] .- V[1:(N-1)])/dk
    dVb[1] = (y[1] - δ*kmin)^(-γ) # state constraint, for stability

    # consumption and savings with forward difference
    cf = max.(dVf,eps()).^(-1/γ)
    muf = y - δ.*k - cf
    Hf = (cf.^(1-γ))/(1-γ) + dVf.*muf

    # consumption and savings with backward difference
    cb = max.(dVb,eps()).^(-1/γ)
    mub = y - δ.*k - cb
    Hb = (cb.^(1-γ))/(1-γ) + dVb.*mub

    # consumption and derivative of value function at steady state
    c0 = y - δ.*k
    dV0[1:N] = max.(c0, eps()).^(-γ)
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

    distance = maximum(abs.(Vchange))
    dist[iter] = distance

    if distance < crit
        if !silent
            println("Value Function Converged, Iteration = ", iter)
        end
        dist[iter+1:end] .= distance
        value = Value(
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

    value = Value(
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
            init_value.v[:] = value.(res.value.v)
            init_value.dVf[:] = value.(res.value.dVf)
            init_value.dVb[:] = value.(res.value.dVb)
            init_value.dV0[:] = value.(res.value.dV0)
    end
end



function solve_HJB(m::Model, hyperparams::HyperParams, state::StateSpace; init_value = Value(hyperparams), maxit = 1000)
    curr_iter = 0
    val = deepcopy(init_value)
    for n in 1:maxit
        curr_iter += 1
        output_value, curr_iter = update_v(m, val, state, hyperparams, iter = n)
        if output_value.convergence_status
            fit_value, _, fit_variables = update_v(m, val, state, hyperparams, iter = curr_iter, silent = true)
            return (value = fit_value, variables = fit_variables, iter = curr_iter)
            break
        end
        val = output_value
    end
    return (value = val, variables = nothing, iter = curr_iter)
end

function solve_HJB(m::Model, hyperparams::HyperParams; init_value = Value(hyperparams), maxit = 1000)
    state = StateSpace(m, hyperparams)
    return solve_HJB(m, hyperparams, state; init_value = init_value, maxit = maxit)
end





dV_Upwind(value::Value, variables::NamedTuple) = value.dVf .* variables.If .+ value.dVb .* variables.Ib .+ value.dV0 .* variables.I0
V_err(m::Union{SkibaModel,SmoothSkibaModel,RamseyCassKoopmansModel}) = (value::Value, variables::NamedTuple) -> variables.c .^ (1-m.γ) / (1-m.γ) .+ dV_Upwind(value, variables) .* k_dot(m)(variables) .- m.ρ .* value.v
