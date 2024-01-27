# This is a fixed point iteration loop to solve the Hamilton Jacobi BellmanPDE
# for the Neoclassical Growth Model with a convex-concave production function
# as in Skiba (1978) "Optimal Growth with a Convex-Concave Production Function"
# Use "butterfly technology":
# f(k) = max(f_H(k),f_L(k)), f_H(k) = A_H*max(k - kappa,0).^a, f_L(k) = A_L*k^a;
# Matlab code originally written by Greg Kaplan and Benjamin Moll - thanks fellas


struct SkibaModel <: Model
    γ::Float64
    α::Float64
    ρ::Float64
    δ::Float64
    A_H::Float64
    A_L::Float64
    κ::Float64
end

#### Util Functions ####
# Util functions to dispatch on for Skiba models
# Create a HyperParams object from a SkibaModel
# use high steady state to guide grid formation
function HyperParams(m::SkibaModel; N = 1000, kmax_f = 1.3, kmin_f = 0.001)
    kssH = k_steady_state_hi(m)
    kmin, kmax = kmin_f*kssH, kmax_f*kssH
    dk = (kmax-kmin)/(N-1)
    HyperParams(N, dk, kmax, kmin)
end

function StateSpace(m::SkibaModel, hyperparams::HyperParams)
    k = range(hyperparams.kmin, hyperparams.kmax, length = hyperparams.N)
    y = production_function(m).(k)
    StateSpace((k = k, y = y))
end



function SkibaModel(; γ = 2.0, α = 0.3, ρ = 0.05, δ = 0.05, A_H = 0.6, A_L = 0.4, κ = 2.0)
    SkibaModel(γ, α, ρ, δ, A_H, A_L, κ)
end




k_steady_state_hi(m::SkibaModel) = (m.α*m.A_H/(m.ρ + m.δ))^(1/(1-m.α)) + m.κ
k_steady_state_lo(m::SkibaModel) = (m.α*m.A_L/(m.ρ + m.δ))^(1/(1-m.α))
k_star(m::SkibaModel) = m.κ/(1-(m.A_L/m.A_H)^(1/m.α))


y_H(m::SkibaModel) = (k) -> m.A_H*max(k - m.κ,0)^m.α
y_L(m::SkibaModel) = (k) -> m.A_L*k^m.α 
production_function(m::SkibaModel) = (k) -> max(NaNMath.pow(m.A_H*max(k - m.κ,0), m.α), NaNMath.pow(m.A_L*k, m.α))

function update_v(m::SkibaModel, value::Value, state::StateSpace, hyperparams::HyperParams; iter = 0, crit = 10^(-6), Delta = 1000, silent = false)
    (; γ, α, ρ, δ, A_H, A_L, κ ) = m
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
        
    if maximum(abs.(sum(A, dims=2))) > 10^(-12)
        stop(println("Improper Transition Matrix"))
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








#### Misc Functions ####
k_dot(m::SkibaModel) = (variables::NamedTuple) -> variables.y .- m.δ .* variables.k .- variables.c
dV_Upwind(value::Value, variables::NamedTuple) = value.dVf .* variables.If .+ value.dVb .* variables.Ib .+ value.dV0 .* variables.I0
V_err(m::SkibaModel) = (value::Value, variables::NamedTuple) -> variables.c .^ (1-m.γ) / (1-m.γ) .+ dV_Upwind(value, variables) .* k_dot(m)(variables) .- m.ρ .* value.v

#### Plotting ####
function plot_production_function(m::SkibaModel, k)
    y = production_function(m).(k)
    yH = y_H(m).(k)
    yL = y_L(m).(k)
    plot(k, y, label="y")
    plot!(k, yH, linestyle=:dash, label="yH")
    plot!(k, yL, linestyle=:dash, label="yL")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end

