"""
    Evolution(fit::GrowthSolution; dt)
    Evolution(fit::DiscreteGrowthSolution)

Cached distribution propagation for a converged fit, used with `trajectory`,
`advance` and `observe`. Continuous-time fits require `dt` and return a
`ContinuousEvolution`. Discrete-time fits advance by integer periods, take no `dt`,
and return a `DiscreteEvolution`.
"""
abstract type Evolution end

"""Private cached integration of a converged continuous-time fit.
One object per concurrent simulation. Observation times never change its fixed
integration lattice or implicit-Euler factorization.
"""
struct ContinuousEvolution{G,A,S,T} <: Evolution
    grid::G
    generator::A
    solver::S
    dt::T
    function ContinuousEvolution(::OwnedSnapshot,g,a,s,dt)
        new{typeof(g),typeof(a),typeof(s),typeof(dt)}(g,a,s,dt)
    end
end
Base.propertynames(::ContinuousEvolution) = (:grid,:dt)
function Base.getproperty(e::ContinuousEvolution,key::Symbol)
    key in (:grid,:dt) || throw(ArgumentError("Evolution exposes only grid and dt; numerical storage is private"))
    return getfield(e,key)
end
function Evolution(fit::GrowthSolution; dt)
    require_converged(fit)
    dt isa AbstractFloat || dt isa Integer || throw(ArgumentError("dt must be a real numerical integration setting"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    A = forward_generator(fit)
    system = sparse(I - dt*A)
    preferred = preferred_blocksize(fit.grid)
    solver = SparseBlockSolver(system; blocksize=independent_blocksize(system,preferred))
    return ContinuousEvolution(OWNED, deepcopy(fit.grid), copy(A), solver, float(dt))
end

struct Trajectory{G,T,M,S,D}
    grid::G
    times::T
    probabilities::M
    settings::S
    diagnostics::D
    function Trajectory(::OwnedSnapshot,g,t,p,s,d)
        new{typeof(g),typeof(t),typeof(p),typeof(s),typeof(d)}(g,t,p,s,d)
    end
end
struct Observations{T,V,S,D}
    times::T
    values::V
    settings::S
    diagnostics::D
    function Observations(::OwnedSnapshot,t,v,s,d)
        new{typeof(t),typeof(v),typeof(s),typeof(d)}(t,v,s,d)
    end
end
Base.getproperty(path::Union{Trajectory,Observations},key::Symbol) = _readonly(getfield(path,key))
function observation_index(path, t)
    i = searchsortedfirst(path.times,t)
    i <= length(path.times) && path.times[i] == t || throw(ArgumentError("Time $t was not requested"))
    return i
end
function at(path::Trajectory, t)
    i = observation_index(path,t)
    return distribution(path.grid; masses=reshape(copy(view(path.probabilities,:,i)),size(path.grid)))
end
mass(path::Trajectory,t) = mass(at(path,t))
density(path::Trajectory,t) = density(at(path,t))
final(path::Trajectory) = at(path,last(path.times))
Base.getindex(path::Trajectory,i::Integer) = at(path,path.times[i])
function expectation(path::Trajectory,f)
    weights = [f(grid_state(path.grid,i)) for i in CartesianIndices(size(path.grid))]
    return [sum(vec(weights) .* view(path.probabilities,:,j)) for j in eachindex(path.times)]
end

function observation_times(times,t0)
    isfinite(t0) || throw(ArgumentError("t0 must be finite"))
    t = Float64.(collect(times))
    !isempty(t) && all(isfinite,t) && all(diff(t) .> 0) && first(t) >= t0 ||
        throw(ArgumentError("Observation times must be nonempty, finite, increasing, and at or after t0"))
    return t
end

function walk_observations(emit, e::ContinuousEvolution, initial::StateDistribution, times, t0)
    compatible_grid(e.grid, initial.grid) || throw(ArgumentError("Initial distribution grid differs from evolution grid"))
    validate_probability(getfield(initial,:probabilities))
    T = promote_type(eltype(getfield(e,:generator)),eltype(getfield(initial,:probabilities)))
    lower = Vector{T}(vec(getfield(initial,:probabilities)))
    upper = similar(lower)
    observed = similar(lower)
    lattice_index = 0
    upper_ready = false
    steps = 0
    max_mass_error = abs(primal(sum(lower))-1)
    for (j,t) in enumerate(times)
        q = (t-t0)/e.dt
        nearest = round(q)
        # Only remove roundoff in a lattice coordinate; real off-lattice times
        # are interpolated. The origin is independent of the output schedule.
        abs(q-nearest) <= 8eps(max(abs(q),1.0)) && (q=nearest)
        target = floor(Int,q)
        while lattice_index < target
            if !upper_ready
                solve_linear!(upper,getfield(e,:solver),lower)
                steps += 1
                validate_probability(upper)
                max_mass_error = max(max_mass_error,abs(primal(sum(upper))-1))
            end
            lower, upper = upper, lower
            upper_ready = false
            lattice_index += 1
        end
        fraction = q-target
        if fraction == 0
            copyto!(observed,lower)
        else
            if !upper_ready
                solve_linear!(upper,getfield(e,:solver),lower)
                steps += 1
                validate_probability(upper)
                max_mass_error = max(max_mass_error,abs(primal(sum(upper))-1))
                upper_ready = true
            end
            @. observed = (1-fraction)*lower + fraction*upper
        end
        emit(j,observed)
    end
    return (steps=steps, max_mass_error=max_mass_error)
end
integration_settings(e,t0) = (dt=e.dt,t0=float(t0),algorithm=:implicit_euler,interpolation=:linear_mass)

"""Store distributions at all requested times. Lookup is only at saved times."""
function trajectory(e::ContinuousEvolution, initial::StateDistribution, times; t0=0.0)
    t = observation_times(times,t0)
    T = promote_type(eltype(getfield(e,:generator)),eltype(getfield(initial,:probabilities)))
    saved = Matrix{T}(undef,length(e.grid),length(t))
    diagnostic = walk_observations(e,initial,t,t0) do j,p
        copyto!(view(saved,:,j),p)
    end
    return Trajectory(OWNED, deepcopy(e.grid),t,saved,integration_settings(e,t0),diagnostic)
end

"""Return only the final distribution after a duration, without retaining history.
This is a fresh one-shot integration, not incremental numerical continuation.
"""
function advance(e::ContinuousEvolution, initial::StateDistribution, duration; t0=0.0)
    isfinite(duration) && duration >= 0 || throw(ArgumentError("Duration must be finite and nonnegative"))
    times = observation_times([t0+duration],t0)
    T = promote_type(eltype(getfield(e,:generator)),eltype(getfield(initial,:probabilities)))
    saved = Vector{T}(undef,length(e.grid))
    walk_observations(e,initial,times,t0) do _,p
        copyto!(saved,p)
    end
    return distribution(e.grid;masses=reshape(saved,size(e.grid)))
end

"""Stream named expectations of named-state functions without storing distributions."""
function observe(e::ContinuousEvolution, initial::StateDistribution, times, observables::NamedTuple; t0=0.0)
    isempty(observables) && throw(ArgumentError("Supply at least one named observable"))
    t = observation_times(times,t0)
    weights = map(f -> vec([f(grid_state(e.grid,i)) for i in CartesianIndices(size(e.grid))]),observables)
    scalar = promote_type(eltype(getfield(e,:generator)),eltype(getfield(initial,:probabilities)),map(eltype,values(weights))...)
    result = map(_ -> Vector{scalar}(undef,length(t)),weights)
    diagnostic = walk_observations(e,initial,t,t0) do j,p
        for key in keys(weights)
            result[key][j] = sum(weights[key] .* p)
        end
    end
    return Observations(OWNED, t,result,integration_settings(e,t0),diagnostic)
end

"""Limiting probabilities, preserving explicit initial class shares. Without an
initial distribution, population shares are used when each positive-share type
has a unique closed class; otherwise an initial distribution is required. No dt.
"""
function stationary(fit::GrowthSolution, initial::Union{Nothing,StateDistribution}=nothing)
    require_converged(fit)
    if initial !== nothing
        compatible_grid(fit.grid,initial.grid) || throw(ArgumentError("Stationary initial grid differs from fit grid"))
        validate_probability(getfield(initial,:probabilities))
    end
    A = forward_generator(fit)
    T = initial === nothing ? eltype(A) : promote_type(eltype(A),eltype(getfield(initial,:probabilities)))
    p = if initial !== nothing
        stationary_mass(SparseMatrixCSC{T,Int}(A);initial_mass=vec(getfield(initial,:probabilities)))
    elseif population(getfield(fit,:model)) !== nothing
        population_stationary_mass(A,fit.grid,population(getfield(fit,:model)))
    else
        stationary_mass(A)
    end
    return distribution(fit.grid;masses=reshape(p,size(fit.grid)))
end
