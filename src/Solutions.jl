"""A finite HJB iterate, policy, diagnostics, and reproducible solve inputs.
Raw arrays are inspectable even when `converged` is false. Economic evaluators
and probability evolution require convergence.
"""
struct GrowthSolution{M,G,V,P,D,S,A}
    model::M
    grid::G
    value::V
    policy::P
    converged::Bool
    diagnostics::D
    settings::S
    initial_value::V
    generator::A
    function GrowthSolution(::OwnedSnapshot,m,g,v,p,c,d,s,initial,a)
        new{typeof(m),typeof(g),typeof(v),typeof(p),typeof(d),typeof(s),typeof(a)}(m,g,v,p,c,d,s,initial,a)
    end
end
Base.propertynames(::GrowthSolution) = (:model,:grid,:value,:policy,:converged,:diagnostics,:settings,:initial_value,:generator)
function Base.getproperty(fit::GrowthSolution,key::Symbol)
    key == :model && return deepcopy(getfield(fit,:model))
    return _readonly(getfield(fit,key))
end
struct ValueGuess{G,V}
    grid::G
    value::V
    function ValueGuess(::OwnedSnapshot,g,v)
        new{typeof(g),typeof(v)}(g,v)
    end
end
Base.getproperty(guess::ValueGuess,key::Symbol) = _readonly(getfield(guess,key))

function require_converged(fit::GrowthSolution)
    fit.converged || throw(ArgumentError("HJB did not converge: $(fit.diagnostics.stop_reason), residual=$(fit.diagnostics.residual_norm) after $(fit.diagnostics.iterations) iterations"))
    return fit
end

# Warm guesses are initialization, not a dependency on a previous solve's Dual
# seed. Strip all old tangents and promote into the current numerical scalar type.
guess_values(x::AbstractArray) = primal.(x)
function initial_values(initial, g)
    if initial isa Union{GrowthSolution,ValueGuess}
        compatible_grid(initial.grid, g) || throw(ArgumentError("Warm-start grid differs; use regrid explicitly"))
        return guess_values(initial.value), :warm
    elseif initial isa AbstractArray
        size(initial) == size(g) || throw(DimensionMismatch("Initial values must have the grid shape"))
        return guess_values(initial), :custom
    elseif initial === nothing
        return nothing, :default
    end
    throw(ArgumentError("initial must be a solution, regridded guess, shaped array, or nothing"))
end

function solution_result(m, g, ws, v, initial, settings, iterations, stop, update, residual, history)
    # Workspace is private to this call; returned arrays have no aliases into
    # other fits or the supplied initialization.
    policy = (consumption=copy(ws.c), drift=ws.c0 .- ws.c, resources=copy(ws.y))
    boundary = (lower_drift=copy(selectdim(policy.drift, 1, 1)),
                upper_drift=copy(selectdim(policy.drift, 1, size(g)[1])))
    diagnostic = (stop_reason=stop, iterations=iterations, update_norm=update,
        residual_norm=residual, admissible=admissible_hjb_policy(ws),
        history=history, boundary=boundary)
    return GrowthSolution(OWNED, deepcopy(m), deepcopy(g), copy(v), policy, stop == :converged,
        diagnostic, settings, copy(initial), copy(ws.A))
end

function nonlinear_residual(ws, m, v)
    residual = ws.A * vec(v)
    for i in eachindex(residual)
        residual[i] += utility(ws.c[i], preferences(m).γ) - preferences(m).ρ * v[i]
    end
    return maximum(convergence_norm, residual)
end

"""
    solve(model, grid; initial=nothing, maxiters=1000, value_tolerance=1e-6,
          residual_tolerance=1e-6, Delta=nothing, history=false)

Solve on the supplied fixed grid. A different grid requires explicit regridding
of a warm start. Both value and tangent changes and nonlinear residuals must
converge. All resolved settings and the promoted starting array are retained.
A numerical failure returns the last finite iterate; invalid inputs or absence
of any finite iterate throw. No public linear-solver strategy is required.
"""
function solve(m::Model, grid::StateGrid; initial=nothing, maxiters=1000,
        value_tolerance=1e-6, residual_tolerance=1e-6, Delta=nothing, history=false)
    validate_utility(m)
    validate_model_grid(m, grid)
    maxiters isa Integer && maxiters >= 0 || throw(ArgumentError("maxiters must be a nonnegative integer"))
    all(x -> isfinite(x) && x > 0, (value_tolerance, residual_tolerance, preferences(m).ρ)) || throw(ArgumentError("Tolerances and discount rate must be finite and positive"))
    Delta = Delta === nothing ? (preferences(m).γ < 1 ? 1.0 : 1000.0) : Delta
    isfinite(Delta) && Delta > 0 || throw(ArgumentError("Delta must be finite and positive"))
    g = deepcopy(grid)
    guess, strategy = initial_values(initial, g)
    value_type = guess === nothing ? Float64 : eltype(guess)
    ws = HJBWorkspace(m, g; value_type=value_type)
    v = similar(ws.next_v)
    v .= guess === nothing ? initial_guess(m, ws.y) : guess
    all(isfinite, v) || throw(ArgumentError("No finite initial value is available"))
    initial_snapshot = copy(v)
    settings = (value_tolerance=value_tolerance, residual_tolerance=residual_tolerance,
        maxiters=maxiters, Delta=Delta, initialization=strategy,
        algorithm=:implicit_upwind_hjb, history=history)
    trace = NamedTuple{(:iteration,:update,:residual),Tuple{Int,Float64,Float64}}[]
    # Prepare once, then retain the candidate's policy for the next iteration.
    try
        prepare_hjb_step!(ws, m, v, Delta)
    catch err
        err isa Union{ValueFunctionError,DomainError} || rethrow()
        throw(ArgumentError("No finite initial value/policy pair is available: $(sprint(showerror,err))"))
    end
    residual = nonlinear_residual(ws, m, v)
    isfinite(residual) || throw(ArgumentError("No finite initial value/policy residual is available"))
    if maxiters == 0
        return solution_result(m,g,ws,v,initial_snapshot,settings,0,:iteration_limit,Inf,residual,trace)
    end
    solver = is_deterministic(m) ? TridiagonalHJBSolver(ws.B) :
        SparseBlockSolver(ws.B; blocksize=independent_blocksize(ws.B,ws.blocksize))
    update = Inf
    for n in 1:maxiters
        try
            n > 1 && refactor!(solver, ws.B)
            solve_linear!(vec(ws.next_v), solver, ws.rhs)
            update = hjb_distance(ws, v)
            prepare_hjb_step!(ws, m, ws.next_v, Delta)
            candidate_residual = nonlinear_residual(ws, m, ws.next_v)
            isfinite(candidate_residual) || throw(ValueFunctionError("Nonfinite candidate residual"))
            copyto!(v, ws.next_v)
            residual = candidate_residual
        catch err
            if err isa Union{ValueFunctionError,DomainError,SingularException}
                prepare_hjb_step!(ws, m, v, Delta)
                residual = nonlinear_residual(ws, m, v)
                return solution_result(m,g,ws,v,initial_snapshot,settings,n,:nonfinite,update,residual,trace)
            end
            rethrow()
        end
        history && push!(trace, (iteration=n, update=Float64(update), residual=Float64(residual)))
        if update < value_tolerance && residual < residual_tolerance && admissible_hjb_policy(ws)
            return solution_result(m,g,ws,v,initial_snapshot,settings,n,:converged,update,residual,trace)
        end
    end
    return solution_result(m,g,ws,v,initial_snapshot,settings,maxiters,:iteration_limit,update,residual,trace)
end

function consumption(fit::GrowthSolution, state::NamedTuple)
    require_converged(fit)
    return interpolate_grid(getfield(fit,:policy).consumption, fit.grid, state)
end
function capital_drift(fit::GrowthSolution,state::NamedTuple)
    require_converged(fit)
    m=getfield(fit,:model)
    return resources(m,state)-preferences(m).δ*state.k-consumption(fit,state)
end
function regrid(fit::GrowthSolution, grid::StateGrid; extrapolate=false)
    validate_regrid(fit.grid,grid)
    source = primal.(fit.value)
    values = map(i -> interpolate_grid(source, fit.grid, grid_state(grid,i); extrapolate=extrapolate), CartesianIndices(size(grid)))
    return ValueGuess(OWNED, deepcopy(grid), values)
end
