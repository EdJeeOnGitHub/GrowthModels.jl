

struct PoissonProcess{T <: Real} <: StochasticProcess
    z::AbstractVector{T}
    Q::AbstractMatrix{T}
end
function PoissonProcess(; z::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    T = promote_type(eltype(z), eltype(Q))
    PoissonProcess(T.(z), T.(Q))
end

function process_mean(p::PoissonProcess)
    (; z, Q) = p
    π = nullspace(Q')[:, 1]
    π = π / sum(π)
    return dot(π, z)
end




struct OrnsteinUhlenbeckProcess{T <: Real} <: StochasticProcess  
    # y = log(z) ~ N(0, ln_stationary_σ)
    # y ~ N(ln_mean, ln_stationary_σ)

    # dlog(z) = -θ log(z) dt + σ dW
    # lognormal variance = exp(μ + σ^2 / 2) => exp(σ^2 / (2*θ) / 2 )
    # ln_stationary_σ = σ^2 / (2θ)

    θ::T  # Rate of mean reversion
    σ::T  # Volatility
    ρ::T  # autocorr
    ln_stationary_μ::T  # Mean of lognormal 
end

function OrnsteinUhlenbeckProcess(;θ::Real, σ::Real)
    θ, σ = promote(θ, σ)
    ln_stationary_μ = exp((σ^2)/(2*θ)/2)
    ρ = exp(-θ)
    OrnsteinUhlenbeckProcess(θ, σ, ρ, ln_stationary_μ)
end

function from_stationary_OrnsteinUhlenbeckProcess(; ρ::Float64, ln_stationary_μ::Float64)
    θ = -log(ρ)
    σ = 2 * sqrt(θ * log(ln_stationary_μ))
    OrnsteinUhlenbeckProcess(θ, σ, ρ, ln_stationary_μ)
end



# Adjusting the function to return the mean of the process correctly
process_mean(p::OrnsteinUhlenbeckProcess) = p.ln_stationary_μ


"""
    sample(process::OrnsteinUhlenbeckProcess, x0, T, dt; seed=nothing)

Sample a trajectory from the Ornstein-Uhlenbeck process starting from `x0` at time `t=0` until time `T` with time step `dt`.

# Arguments
- `process`: Ornstein-Uhlenbeck process instance.
- `x0`: Initial value of the process.
- `T`: Total time of simulation.
- `dt`: Time step.
- `seed`: Optional seed for random number generator.

# Returns
- `times`: Array of times at which the process is sampled.
- `values`: Sampled values of the process.
# Example usage
θ = 1.0
σ = 2.0
ou_process = OrnsteinUhlenbeckProcess(θ=θ, σ=σ)
x0 = [0.0, 0.01]  # Initial value
T = 100.0  # Total time
dt = 1.0  # Time step

times, values = sample(ou_process, x0, T, dt)
"""
function sample(process::OrnsteinUhlenbeckProcess, x0::Float64, T, dt; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    T >= 0 && dt > 0 || throw(ArgumentError("T must be nonnegative and dt positive"))
    times = 0:dt:T
    N = length(times) - 1
    values = zeros(N+1)
    values[1] = x0

    # Exact solution parameters
    ρ = exp(-process.θ * dt)
    # The variance of the noise term in the exact solution
    noise_std = process.σ * sqrt((1 - exp(-2 * process.θ * dt)) / (2 * process.θ))

    for i in 2:N+1
        # Exact update rule
        values[i] = values[i-1] * ρ + noise_std * randn()
    end
  
    values = exp.(values)
    return times, values
end


"""
    sample(process::OrnsteinUhlenbeckProcess, x0s, T, dt; seed=nothing)

Sample trajectories from the Ornstein-Uhlenbeck process starting from multiple initial conditions `x0s` at time `t=0` until time `T` with time step `dt`.

# Arguments
- `process`: Ornstein-Uhlenbeck process instance.
- `x0s`: Array of initial values of the process for each trajectory.
- `T`: Total time of simulation.
- `dt`: Time step.
- `seed`: Optional seed for random number generator.

# Returns
- `times`: Array of times at which the process is sampled.
- `values`: Sampled values of the process for each initial condition. Each row corresponds to a trajectory starting from the respective initial condition in `x0s`.
# Example usage
θ = 1.0
σ = 2.0
ou_process = OrnsteinUhlenbeckProcess(θ=θ, σ=σ)
x0s = [0.0, 0.5, -0.5, 1.0, -1.0]  # Initial values for multiple trajectories
T = 100.0  # Total time
dt = 1.0  # Time step

times, values = sample(ou_process, x0s, T, dt)

# `times` is an array of sample times
# `values` is a matrix where each row corresponds to the trajectory of the process starting from each initial condition in `x0s`

plot(times, values')

"""
function sample(process::OrnsteinUhlenbeckProcess, x0s::Vector, T, dt; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    N_traj = length(x0s)
    T >= 0 && dt > 0 || throw(ArgumentError("T must be nonnegative and dt positive"))
    times = 0:dt:T
    N_time = length(times) - 1
    values = zeros(N_traj, N_time+1)
    
    # Exact solution parameters
    ρ = exp(-process.θ * dt)
    noise_std = process.σ * sqrt((1 - exp(-2 * process.θ * dt)) / (2 * process.θ))

    for n in 1:N_traj
        values[n, 1] = x0s[n]
        for i in 2:N_time+1
            # Exact update rule
            values[n, i] = values[n, i-1] * ρ + noise_std * randn()
        end
    end
    values = exp.(values)
    return times, values
end
"""
    sample(process::OrnsteinUhlenbeckProcess, x0, T, dt, K; seed=nothing)

Sample trajectories from the Ornstein-Uhlenbeck process starting from an N-dimensional initial condition `x0`, K times, over a time period `T` with time step `dt`.

# Arguments
- `process`: Ornstein-Uhlenbeck process instance.
- `x0`: N-dimensional initial value of the process.
- `T`: Total time of simulation.
- `dt`: Time step.
- `K`: Number of trajectories to sample.
- `seed`: Optional seed for random number generator.

# Returns
- `times`: Array of times at which the process is sampled.
- `values`: 3D array of sampled values. Dimensions are [Dimension of x0, Time Steps, Number of Trajectories].
# Example usage
θ = 1.0
σ = 2.0
ou_process = OrnsteinUhlenbeckProcess(θ=θ, σ=σ)
x0 = [0.0, 0.5]  # 2-dimensional initial condition
T = 10.0  # Total time
dt = 0.01  # Time step
K = 100  # Number of trajectories

times, values = sample(ou_process, x0, T, dt, K)

# `times` is an array of sample times
# `values` is a 3D array where dimensions are [Dimension of x0, Time Steps, Number of Trajectories]
"""
function sample(process::OrnsteinUhlenbeckProcess, x0, T, dt, K; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    N_dim = length(x0)
    T >= 0 && dt > 0 || throw(ArgumentError("T must be nonnegative and dt positive"))
    times = 0:dt:T
    N_time = length(times) - 1
    values = zeros(N_dim, N_time+1, K)
    
    # Exact solution parameters
    ρ = exp(-process.θ * dt)
    noise_std = process.σ * sqrt((1 - exp(-2 * process.θ * dt)) / (2 * process.θ))

    for k in 1:K
        values[:, 1, k] = x0
        for i in 2:N_time+1
            dw = randn(N_dim)
            # Exact update rule for each dimension
            values[:, i, k] = values[:, i-1, k] .* ρ .+ noise_std .* dw
        end
    end
    values = exp.(values)
    return times, values
end





"""
    generator(process, coordinates)

Return a sparse continuous-time generator on the process axis, acting on value
functions (rows sum to zero, off-diagonal entries are nonnegative). The solver
embeds it in the joint state layout. Custom processes implement this method and
use explicit grid coordinates or model-provided bounds.
"""
function generator(p::PoissonProcess, z::AbstractVector)
    z == p.z || throw(ArgumentError("Poisson coordinates must equal process support"))
    Q = sparse(p.Q)
    size(Q) == (length(z),length(z)) || throw(DimensionMismatch("Poisson generator size differs from support"))
    check_generator(Q)
    return copy(Q)
end
function generator(p::OrnsteinUhlenbeckProcess, z::AbstractVector)
    length(z) >= 2 && all(isfinite,z) && first(z) > 0 && all(diff(z) .> 0) ||
        throw(ArgumentError("OU coordinates must be positive and increasing"))
    dz = (last(z)-first(z))/(length(z)-1)
    maximum(abs.(diff(z) .- dz)) <= 1e-6 || throw(ArgumentError("OU coordinates must be uniformly spaced"))
    mu = (-p.θ .* log.(z) .+ p.σ^2/2) .* z
    variance = p.σ^2 .* z.^2
    backward = variance ./ (2dz^2) .+ max.(-mu,0) ./ dz
    forward = variance ./ (2dz^2) .+ max.(mu,0) ./ dz
    diagonal = .-(backward .+ forward)
    diagonal[1] += backward[1]
    diagonal[end] += forward[end]
    Q = spdiagm(-1=>backward[2:end],0=>diagonal,1=>forward[1:end-1])
    check_generator(Q)
    return Q
end

"""
    process_axis(process, specification)

Resolve an axis specification (coordinates or a NamedTuple of numerical options)
to `(coordinates=..., kind=:interpolable)` or `kind=:categorical`. Custom processes
implement this together with `generator`. The process owns support restrictions
and option validation; the grid owns axis order and joint layout.
"""
function process_axis end

function validate_options(spec::NamedTuple, allowed, context)
    unknown = setdiff(keys(spec), allowed)
    isempty(unknown) || throw(ArgumentError("Unknown $context options $(Tuple(unknown)); supported options: $allowed"))
    return spec
end
function process_axis(p::PoissonProcess, spec)
    if spec isa NamedTuple
        validate_options(spec, (:n,), "Poisson axis")
        get(spec,:n,length(p.z)) == length(p.z) || throw(DimensionMismatch("Poisson grid size must match the process"))
    else
        primal.(collect(spec)) == primal.(p.z) || throw(ArgumentError("Poisson support belongs to the process"))
    end
    return (coordinates=copy(p.z),kind=:categorical)
end
function process_axis(p::OrnsteinUhlenbeckProcess, spec)
    z = if spec isa NamedTuple
        validate_options(spec, (:n,:bounds,:width), "OU axis")
        n = get(spec,:n,40)
        n isa Integer && n >= 2 || throw(ArgumentError("OU grid needs at least two points"))
        bounds = haskey(spec,:bounds) ? spec.bounds : productivity_bounds(p,get(spec,:width,4))
        collect(range(bounds[1],bounds[2];length=n))
    else
        collect(spec)
    end
    length(z) >= 2 && all(isfinite,z) && first(z) > 0 && all(diff(z) .> 0) ||
        throw(ArgumentError("OU coordinates must be positive and increasing"))
    dz=(last(z)-first(z))/(length(z)-1)
    maximum(abs.(diff(z).-dz)) <= 1e-6 || throw(ArgumentError("OU coordinates must be uniformly spaced"))
    return (coordinates=z,kind=:interpolable)
end

# Bounds for log productivity use the stationary log standard deviation.
function productivity_bounds(p::OrnsteinUhlenbeckProcess, standard_deviations = 4)
    p.θ > 0 && p.σ >= 0 || throw(ArgumentError("OU grid requires positive mean reversion and nonnegative volatility"))
    standard_deviations > 0 || throw(ArgumentError("z_width must be positive"))
    width = standard_deviations * p.σ / sqrt(2p.θ)
    return exp(-width), exp(width)
end

