# The pecuniary externality of wealth on a credit-market transition path.
#
# Welfare W = ∫ V_0 dμ_0. Moving a household changes the distribution, hence excess demand
# and the clearing rate at each date, hence everyone's income through r. To first order:
#   dW/dr_t = dt e^{-ρt} ∫ u'(c_t) ∂y_t/∂r dμ_t,      ∂y/∂r = a - k (envelope; 0 if the floor binds)
#   dr_t     = -(g_t · dμ_t) / E_r(t),                g_t(s) = k(s, r_t) - a, E_r = ∂K/∂r
#   dμ_t     = M_{t-1} ⋯ M_1 dμ_1                     (forward steps of the transition)
# so the welfare value of a unit of mass at state s at t=0, through rates, is
#   φ(s) = Σ_t w_t [(M_{t-1} ⋯ M_1)' g_t](s),         w_t = -(dW/dr_t) / E_r(t),
# computed by one backward (adjoint) recursion. The rate response is the static clearing
# response at each date, holding savings policies at the path's: feedback from the rate path
# to later wealth is not included.

"""
    rate_externality(model, grid, path::CreditTransition; rate_step=1e-6)

Welfare value at `t=0` of one unit of mass at each state of `grid`, through its effect on the
equilibrium rate path of `path`, in the same units as `path.value` integrated against a
distribution. Returns `(φ, dW_dr, demand_slope, weights)`: `φ` has the shape of `grid`; the
others are per date. A planner choosing a redistribution of wealth at `t=0` values moving a
household from `s` to `s′` at `V(s′) - V(s) + φ(s′) - φ(s)` to first order.
"""
function rate_externality(base::Model,grid::StateGrid,path::CreditTransition;rate_step=1e-6)
    compatible_grid(grid,getfield(path,:grid)) || throw(ArgumentError("Grid differs from the path's grid"))
    λ = getfield(path,:λ); rates = collect(getfield(path,:rates)); N = length(rates)
    settings = getfield(getfield(path,:distributions),:settings); dt = settings.dt
    terminal_state = only(getfield(path,:terminal).equilibria)
    masses_path = Array{Float64}(getfield(getfield(path,:distributions),:probabilities))
    cache = CreditPathWorkspace(base,grid,λ,dt,N,clearing_rate(getfield(path,:terminal)),
        Array{Float64}(copy(terminal_state.fit.value)),masses_path[:,1])
    backward_pass!(cache,rates)                       # fills the path's drift at each date
    ρ,δ,γ = primal(preferences(base).ρ),primal(preferences(base).δ),primal(preferences(base).γ)
    states = cache.states; n = length(states)
    dW_dr = zeros(N); slope = zeros(N); weights = zeros(N); g = zeros(n,N)
    Threads.@threads for t in 1:N
        r = rates[t]; m = credit_model_at(base,λ,r); p = view(masses_path,:,t)
        acc = 0.0
        for i in 1:n
            s = states[i]; choice = capital_choice(m,s)
            g[i,t] = choice.capital-s.k
            p[i] > 0 || continue
            raw = choice.profit+r*s.k
            # Income responds to r unless the floor binds.
            responds = credit_income(m.model,raw) != credit_income(m.model,raw+1e-9)
            dy = responds ? s.k-choice.capital : 0.0
            c = resources(m,s)-δ*s.k-cache.drift[i,t]
            acc += p[i]*max(c,eps())^(-γ)*dy
        end
        dW_dr[t] = dt*exp(-ρ*(t-1)*dt)*acc
        slope[t] = (capital_demand(base,λ,r+rate_step,states,p)-capital_demand(base,λ,r-rate_step,states,p))/(2rate_step)
        slope[t] < 0 || throw(ArgumentError("Capital demand does not fall with the rate at date $t"))
        weights[t] = -dW_dr[t]/slope[t]
    end
    # Adjoint of the forward steps: m_{t+1} = (I - dt A_t')^{-1} m_t, so the transpose step
    # solves (I - dt A_t) x = ψ.
    ψ = weights[N] .* g[:,N]
    ws = cache.ws
    for t in N-1:-1:1
        set_generator_from_drift!(ws,view(cache.drift,:,t))
        step = sparse(I,n,n)-dt*ws.A
        ψ = step\ψ
        ψ .+= weights[t] .* view(g,:,t)
    end
    return (φ=reshape(ψ,size(grid)),dW_dr=dW_dr,demand_slope=slope,weights=weights)
end
