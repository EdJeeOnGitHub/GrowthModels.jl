"""
    simulate(fit::DiscreteGrowthSolution, initial_state, periods; rng=Random.default_rng())

Simulate one individual's continuous-capital path under the solved finite-state
productivity chain. Current productivity is observed before choosing capital;
next-period productivity is drawn afterward, while ability remains fixed. Policies
are globally re-optimized at every capital state, without grid lotteries.
Returns owned `states` (length periods+1), `consumption` (length periods), and
integer `times`. Pass an explicit RNG for reproducibility. For LogAR1 this samples
the Rouwenhorst chain, not the underlying continuous Gaussian process.
"""
function simulate(fit::DiscreteGrowthSolution,initial::NamedTuple,periods;rng=Random.default_rng())
    require_converged(fit)
    n=discrete_period(periods)
    n < typemax(Int) || throw(ArgumentError("Path length overflows Int"))
    discrete_state_indices(fit.grid,initial)
    names=keys(getfield(fit.grid,:axes))
    state=NamedTuple{names}(map(key -> Float64(initial[key]),names))
    states=Vector{typeof(state)}(undef,n+1)
    consumed=Vector{Float64}(undef,n)
    states[1]=state
    expected=discrete_expectation(fit)
    P=productivity_transition(getfield(fit,:model),fit.grid)
    for t in 1:n
        choice=discrete_policy(fit,state;expected)
        consumed[t]=choice.consumption
        z,_=discrete_state_indices(fit.grid,state)
        nextstate=merge(state,(k=choice.capital,))
        if :z in names
            draw=rand(rng)
            cumulative=0.0
            zp=size(P,2)
            for j in axes(P,2)
                cumulative+=P[z,j]
                if draw<cumulative
                    zp=j
                    break
                end
            end
            nextstate=merge(nextstate,(z=fit.grid.z[zp],))
        end
        states[t+1]=nextstate
        state=nextstate
    end
    return (times=collect(0:n),states=states,consumption=consumed)
end
