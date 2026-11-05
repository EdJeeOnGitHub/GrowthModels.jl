"""Owned discrete-time value, policy, transition matrix, and Bellman diagnostics.
Arrays are read-only views. Failed fits retain a finite value and its globally
optimized policy; economic evaluators require convergence."""
struct DiscreteGrowthSolution{M,G,V,P,D,S,A}
    model::M
    grid::G
    value::V
    policy::P
    converged::Bool
    diagnostics::D
    settings::S
    initial_value::V
    transition::A
    function DiscreteGrowthSolution(::OwnedSnapshot,m,g,v,p,c,d,s,initial,a)
        new{typeof(m),typeof(g),typeof(v),typeof(p),typeof(d),typeof(s),typeof(a)}(
            m,g,v,p,c,d,s,initial,a)
    end
end
function Base.getproperty(fit::DiscreteGrowthSolution,key::Symbol)
    key == :model && return deepcopy(getfield(fit,key))
    return _readonly(getfield(fit,key))
end
function require_converged(fit::DiscreteGrowthSolution)
    fit.converged || throw(ArgumentError("Bellman solve did not converge: $(fit.diagnostics.stop_reason), residual=$(fit.diagnostics.residual_norm)"))
    return fit
end

function discrete_initial(initial, g)
    if initial isa Union{DiscreteGrowthSolution,ValueGuess}
        compatible_grid(initial.grid,g) || throw(ArgumentError("Warm-start grid differs; use regrid explicitly"))
        raw = initial.value
    elseif initial isa AbstractArray
        raw = initial
    elseif initial === nothing
        return zeros(length(g)), :default
    else
        throw(ArgumentError("Expected a discrete-time solution, ValueGuess, or shaped value array"))
    end
    size(raw) == size(g) || throw(DimensionMismatch("Initial values must have the grid shape"))
    any(x -> x isa ForwardDiff.Dual,raw) && throw(ArgumentError("Discrete solver does not support AD"))
    v = Float64.(raw)
    all(isfinite,v) || throw(ArgumentError("Initial values must be finite"))
    return v, :warm
end

function discrete_result(m,g,v,ws,initial,settings,n,stop,update,residual,history)
    β, γ = preferences(m).discount, preferences(m).risk_aversion
    # Always return policy and residual for the retained value, including failure.
    residual = bellman_step!(ws,g.k,v,β,γ;alternatives=true)
    policy = (consumption=copy(ws.consumption), next_capital=copy(ws.capital),
        resources=[resources(m,(k=x,)) for x in g.k])
    budget_error = maximum(abs, ws.consumption .+ ws.capital .- ws.budget)
    admissible = all(ws.consumption .>= 0) && all(first(g.k) .<= ws.capital .<= last(g.k)) &&
        all(isfinite,discrete_utility.(ws.consumption,γ))
    diagnostics = (stop_reason=stop, iterations=n, update_norm=update,
        residual_norm=residual, value_error_bound=residual/(1-β), budget_error=budget_error,
        admissible=admissible, history=copy(history),
        boundary=(lower_choice=ws.capital .== first(g.k),upper_choice=ws.capital .== last(g.k)),
        alternative_capital=copy(ws.alternative_capital), choice_gap=copy(ws.choice_gap))
    return DiscreteGrowthSolution(OWNED,deepcopy(m),deepcopy(g),copy(v),policy,
        stop == :converged,diagnostics,settings,copy(initial),policy_transition(g.k,ws.capital))
end

function solve_discrete(m,g::StateGrid; initial=nothing,maxiters=2000,
        value_tolerance=1e-8,residual_tolerance=1e-8,
        algorithm=:modified_policy_iteration,policy_steps=20,history=false)
    validate_model_grid(m,g)
    all(x -> !(x isa ForwardDiff.Dual) && isfinite(x) && x >= 0,g.k) ||
        throw(ArgumentError("Discrete capital coordinates must be finite, nonnegative, and non-AD"))
    maxiters isa Integer && maxiters >= 0 || throw(ArgumentError("maxiters must be a nonnegative integer"))
    policy_steps isa Integer && policy_steps >= 0 || throw(ArgumentError("policy_steps must be a nonnegative integer"))
    all(x -> isfinite(x) && x > 0,(value_tolerance,residual_tolerance)) ||
        throw(ArgumentError("Tolerances must be finite and positive"))
    algorithm in (:value_iteration,:modified_policy_iteration) || throw(ArgumentError("Unknown Bellman algorithm"))
    g = deepcopy(g)
    v, strategy = discrete_initial(initial,g)
    initial_snapshot = copy(v)
    ws = BellmanWorkspace(m,g.k)
    β, γ = preferences(m).discount, preferences(m).risk_aversion
    residual = bellman_step!(ws,g.k,v,β,γ)
    isfinite(residual) || throw(ArgumentError("No finite initial Bellman residual"))
    settings = (;algorithm,policy_steps,maxiters,value_tolerance,residual_tolerance,
        initialization=strategy,history,precision=:Float64,tie_rule=:lower_capital)
    trace = NamedTuple{(:iteration,:update,:residual),Tuple{Int,Float64,Float64}}[]
    update, stop, iterations = Inf, :iteration_limit, 0
    for n in 1:maxiters
        iterations = n
        candidate = copy(ws.next_value)
        if algorithm == :modified_policy_iteration
            P = policy_transition(g.k,ws.capital)
            reward = discrete_utility.(ws.consumption,γ)
            for _ in 1:policy_steps
                candidate = reward + β*(P*candidate)
            end
        end
        if !all(isfinite,candidate)
            stop = :nonfinite
            break
        end
        update = maximum(abs,candidate .- v)
        try
            residual = bellman_step!(ws,g.k,candidate,β,γ)
        catch err
            err isa ArgumentError || rethrow()
            stop = :nonfinite
            break
        end
        if !isfinite(residual)
            stop = :nonfinite
            break
        end
        copyto!(v,candidate)
        history && push!(trace,(iteration=n,update=update,residual=residual))
        if update <= value_tolerance && residual <= residual_tolerance
            stop = :converged
            break
        end
    end
    return discrete_result(m,g,v,ws,initial_snapshot,settings,iterations,stop,update,residual,trace)
end

function discrete_policy(fit::DiscreteGrowthSolution,state::NamedTuple)
    require_converged(fit)
    keys(state) == (:k,) || throw(ArgumentError("State must have exactly the k coordinate"))
    state.k isa ForwardDiff.Dual && throw(ArgumentError("Discrete policy queries do not support AD"))
    k = fit.grid.k
    first(k) <= state.k <= last(k) || throw(ArgumentError("State outside capital support"))
    m = getfield(fit,:model)
    v = getfield(fit,:value)
    p = preferences(m)
    return bellman_choice(discrete_budget(m,state.k),k,v,diff(v)./diff(k),p.discount,p.risk_aversion)
end
"""Optimal next-period capital, re-optimized against the fitted value at the query state."""
next_capital(fit::DiscreteGrowthSolution,state::NamedTuple) = discrete_policy(fit,state).capital
consumption(fit::DiscreteGrowthSolution,state::NamedTuple) = discrete_policy(fit,state).consumption
function regrid(fit::DiscreteGrowthSolution,g::StateGrid;extrapolate=false)
    validate_regrid(fit.grid,g)
    v = [interpolate_grid(fit.value,fit.grid,(k=x,);extrapolate) for x in g.k]
    return ValueGuess(OWNED,deepcopy(g),v)
end

"""
    solve(model::DiscreteTimeModel, grid; initial=nothing, maxiters=2000,
        value_tolerance=1e-8, residual_tolerance=1e-8,
        algorithm=:modified_policy_iteration, policy_steps=20, history=false)

Globally maximize each piecewise-linear continuation interval. `:value_iteration`
provides the reference algorithm; modified policy iteration performs `policy_steps`
fixed-policy sweeps between global improvements. Both update and global Bellman
residual must converge. Numerical grid bounds constrain actions; inspect boundary
choices and refine bounds and resolution together. This Float64 solver has no AD
contract. Native discrete resource wrappers are also supported.
"""
solve(m::DiscreteTimeModel,g::StateGrid;kwargs...) = solve_discrete(m,g;kwargs...)
function solve(m::ResourceModel,g::StateGrid;kwargs...)
    base_model(m) isa DiscreteTimeModel && return solve_discrete(m,g;kwargs...)
    return invoke(solve,Tuple{Model,StateGrid},m,g;kwargs...)
end
