"""Owned discrete-time value, policy, transition matrix, and Bellman diagnostics.
Arrays are read-only views. Failed fits retain a finite value and its globally
optimized policy; economic evaluators require convergence."""
struct DiscreteGrowthSolution{M,G,V,P,D,S,A,C}
    model::M
    grid::G
    value::V
    policy::P
    converged::Bool
    diagnostics::D
    settings::S
    initial_value::V
    transition::A
    continuation::C
    function DiscreteGrowthSolution(::OwnedSnapshot,m,g,v,p,c,d,s,initial,a,continuation)
        new{typeof(m),typeof(g),typeof(v),typeof(p),typeof(d),typeof(s),typeof(a),typeof(continuation)}(
            m,g,v,p,c,d,s,initial,a,continuation)
    end
end
Base.propertynames(::DiscreteGrowthSolution) = (:model,:grid,:value,:policy,:converged,:diagnostics,:settings,:initial_value,:transition)
function Base.getproperty(fit::DiscreteGrowthSolution,key::Symbol)
    key == :continuation && throw(ArgumentError("Cached continuation values are private"))
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
    v = vec(Float64.(raw))
    all(isfinite,v) || throw(ArgumentError("Initial values must be finite"))
    return v, :warm
end

function discrete_result(m,g,v,ws,initial,settings,n,stop,update,residual,history)
    β, γ = preferences(m).discount, preferences(m).γ
    # Always return policy and residual for the retained value, including failure.
    residual = bellman_step!(ws,g.k,v,β,γ;alternatives=true)
    shaped(x)=reshape(copy(x),size(g))
    policy=(consumption=shaped(ws.consumption),next_capital=shaped(ws.capital),
        resources=[resources(m,grid_state(g,i)) for i in CartesianIndices(size(g))])
    budget_error = maximum(abs, ws.consumption .+ ws.capital .- ws.budget)
    admissible = all(ws.consumption .>= 0) && all(first(g.k) .<= ws.capital .<= last(g.k)) &&
        all(isfinite,discrete_utility.(ws.consumption,γ))
    diagnostics = (stop_reason=stop, iterations=n, update_norm=update,
        residual_norm=residual, value_error_bound=residual/(1-β), budget_error=budget_error,
        admissible=admissible, history=copy(history),
        boundary=(lower_choice=shaped(ws.capital .== first(g.k)),upper_choice=shaped(ws.capital .== last(g.k)),
            uncovered_steady_states=uncovered_steady_states(m,g)),
        alternative_capital=shaped(ws.alternative_capital), choice_gap=shaped(ws.choice_gap))
    return DiscreteGrowthSolution(OWNED,deepcopy(m),deepcopy(g),shaped(v),policy,
        stop == :converged,diagnostics,settings,shaped(initial),
        policy_transition(g.k,ws.capital,ws.productivity,size(ws.expected,3)),copy(ws.expected))
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
    ws = BellmanWorkspace(m,g)
    β, γ = preferences(m).discount, preferences(m).γ
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
            reward = discrete_utility.(ws.consumption,γ)
            for _ in 1:policy_steps
                candidate = reward + β*policy_action(ws,g.k,candidate)
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

function discrete_state_indices(g,state)
    ax=getfield(g,:axes)
    Set(keys(state)) == Set(keys(ax)) || throw(ArgumentError("State keys must match $(keys(ax))"))
    all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x),values(state)) ||
        throw(ArgumentError("Discrete policy states must be finite and non-AD"))
    first(g.k) <= state.k <= last(g.k) || throw(ArgumentError("State outside capital support"))
    indices=map((:z,:ability)) do name
        !haskey(ax,name) && return 1
        index=findfirst(==(state[name]),ax[name])
        index === nothing && throw(ArgumentError("$name must be a represented categorical state"))
        return index
    end
    return indices
end
discrete_expectation(fit) = getfield(fit,:continuation)
function discrete_policy(fit::DiscreteGrowthSolution,state::NamedTuple;expected=discrete_expectation(fit))
    require_converged(fit)
    z,a=discrete_state_indices(fit.grid,state)
    m=getfield(fit,:model)
    k=fit.grid.k
    continuation=view(expected,:,z,a)
    p=preferences(m)
    return bellman_choice(discrete_budget(m,state),k,continuation,
        diff(continuation)./diff(k),p.discount,p.γ)
end
"""Optimal next-period capital, re-optimized against the fitted value at the query state."""
next_capital(fit::DiscreteGrowthSolution,state::NamedTuple) = discrete_policy(fit,state).capital
consumption(fit::DiscreteGrowthSolution,state::NamedTuple) = discrete_policy(fit,state).consumption
function regrid(fit::DiscreteGrowthSolution,g::StateGrid;extrapolate=false)
    validate_regrid(fit.grid,g)
    v = [interpolate_grid(fit.value,fit.grid,grid_state(g,i);extrapolate) for i in CartesianIndices(size(g))]
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
