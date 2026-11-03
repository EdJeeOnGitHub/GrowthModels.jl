"""Private cached integration of a converged fit. `dt` is required.
One Evolution object per concurrent simulation. Observation times never change
its fixed integration lattice or implicit-Euler factorization.
"""
struct Evolution{G,A,S,T}
    grid::G
    generator::A
    solver::S
    dt::T
end
function Evolution(fit::GrowthSolution; dt)
    require_converged(fit)
    dt isa AbstractFloat || dt isa Integer || throw(ArgumentError("dt must be a real numerical integration setting"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    A = sparse(transpose(fit.generator))
    check_generator(sparse(transpose(A)))
    system = sparse(I - dt*A)
    preferred = length(size(fit.grid)) > 2 ? prod(size(fit.grid)[1:2]) : length(fit.grid)
    solver = SparseBlockSolver(system; blocksize=independent_blocksize(system,preferred))
    return Evolution(deepcopy(fit.grid), copy(A), solver, float(dt))
end

struct Trajectory{G,T,M,S,D}
    grid::G
    times::T
    probabilities::M
    settings::S
    diagnostics::D
end
struct Observations{T,V,S,D}
    times::T
    values::V
    settings::S
    diagnostics::D
end
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

function walk_observations(emit, e::Evolution, initial::StateDistribution, times, t0)
    compatible_grid(e.grid, initial.grid) || throw(ArgumentError("Initial distribution grid differs from evolution grid"))
    validate_probability(getfield(initial,:probabilities))
    T = promote_type(eltype(e.generator),eltype(getfield(initial,:probabilities)))
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
                solve_linear!(upper,e.solver,lower)
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
                solve_linear!(upper,e.solver,lower)
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
function trajectory(e::Evolution, initial::StateDistribution, times; t0=0.0)
    t = observation_times(times,t0)
    T = promote_type(eltype(e.generator),eltype(getfield(initial,:probabilities)))
    saved = Matrix{T}(undef,length(e.grid),length(t))
    diagnostic = walk_observations(e,initial,t,t0) do j,p
        copyto!(view(saved,:,j),p)
    end
    return Trajectory(deepcopy(e.grid),t,saved,integration_settings(e,t0),diagnostic)
end

"""Return only the final distribution after a duration, without retaining history.
This is a fresh one-shot integration, not incremental numerical continuation.
"""
function advance(e::Evolution, initial::StateDistribution, duration; t0=0.0)
    isfinite(duration) && duration >= 0 || throw(ArgumentError("Duration must be finite and nonnegative"))
    times = observation_times([t0+duration],t0)
    T = promote_type(eltype(e.generator),eltype(getfield(initial,:probabilities)))
    saved = Vector{T}(undef,length(e.grid))
    walk_observations(e,initial,times,t0) do _,p
        copyto!(saved,p)
    end
    return distribution(e.grid;masses=reshape(saved,size(e.grid)))
end

"""Stream named expectations of named-state functions without storing distributions."""
function observe(e::Evolution, initial::StateDistribution, times, observables::NamedTuple; t0=0.0)
    isempty(observables) && throw(ArgumentError("Supply at least one named observable"))
    t = observation_times(times,t0)
    weights = map(f -> vec([f(grid_state(e.grid,i)) for i in CartesianIndices(size(e.grid))]),observables)
    scalar = promote_type(eltype(e.generator),eltype(getfield(initial,:probabilities)),map(eltype,values(weights))...)
    result = map(_ -> Vector{scalar}(undef,length(t)),weights)
    diagnostic = walk_observations(e,initial,t,t0) do j,p
        for key in keys(weights)
            result[key][j] = sum(weights[key] .* p)
        end
    end
    return Observations(t,result,integration_settings(e,t0),diagnostic)
end

"""Limiting probability distribution, preserving initial class shares. No dt is used."""
function stationary(fit::GrowthSolution, initial::Union{Nothing,StateDistribution}=nothing)
    require_converged(fit)
    if initial !== nothing
        compatible_grid(fit.grid,initial.grid) || throw(ArgumentError("Stationary initial grid differs from fit grid"))
        validate_probability(getfield(initial,:probabilities))
    end
    A = sparse(transpose(fit.generator))
    T = initial === nothing ? eltype(A) : promote_type(eltype(A),eltype(getfield(initial,:probabilities)))
    p = StationaryDistribution(SparseMatrixCSC{T,Int}(A),ones(T,length(fit.grid));
        initial_density=initial === nothing ? nothing : vec(getfield(initial,:probabilities)))
    return distribution(fit.grid;masses=reshape(p,size(fit.grid)))
end
