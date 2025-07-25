

struct PoissonProcess{T <: Real} <: StochasticProcess
    z::AbstractVector{T}
    Q::AbstractMatrix{T}
end
function PoissonProcess(; z::AbstractVector{T}, Q::AbstractMatrix{T}) where T <: Real
    PoissonProcess(z, Q)
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

    # dlog(z) = -θ log(z) dt + σ^2 dW
    # lognormal variance = exp(μ + σ^2 / 2) => exp(σ^2 / (2*θ) / 2 )
    # ln_stationary_σ = σ^2 / (2θ)

    θ::T  # Rate of mean reversion
    σ::T  # Volatility
    ρ::T  # autocorr
    ln_stationary_μ::T  # Mean of lognormal 
end

function OrnsteinUhlenbeckProcess(;θ::T, σ::T) where T <: Real
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
    N = round(Int, T/dt)
    times = 0:dt:T
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
    N_time = round(Int, T/dt)
    times = 0:dt:T
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
    N_time = round(Int, T/dt)
    times = 0:dt:T
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




