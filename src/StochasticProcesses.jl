
abstract type StochasticProcess end

struct OrnsteinUhlenbeckProcess <: StochasticProcess
    θ
    σ
    ρ
    stationary_σ
    zmean
end

function OrnsteinUhlenbeckProcess(; θ, σ)
    stationary_σ = σ^2/(2*θ)
    ρ = exp(-θ)
    zmean = exp(stationary_σ/2)
    OrnsteinUhlenbeckProcess(θ, σ, ρ, stationary_σ, zmean)
end


function from_stationary_OrnsteinUhlenbeckProcess(; ρ, stationary_σ)
    θ = -log(ρ)
    σ = sqrt(2*θ*stationary_σ) 
    zmean = exp(stationary_σ/2)
    OrnsteinUhlenbeckProcess(θ, σ, ρ, stationary_σ, zmean)
end


OrnsteinUhlenbeckProcess(θ = 1, σ = 2)
from_stationary_OrnsteinUhlenbeckProcess(ρ = 0.3678, stationary_σ = 2.0)
process_mean(p::OrnsteinUhlenbeckProcess) = exp(p.stationary_σ/2) 

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
    N = round(Int, T/dt)  # Number of time steps
    times = 0:dt:T
    values = zeros(N+1)
    values[1] = x0

    for i in 2:N+1
        dw = sqrt(dt) * randn()  # Wiener process increment
        values[i] = values[i-1] + process.θ * (-values[i-1]) * dt + process.σ * dw
    end

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
    N_traj = length(x0s)  # Number of trajectories
    N_time = round(Int, T/dt)  # Number of time steps
    times = 0:dt:T
    values = zeros(N_traj, N_time+1)
    
    for n in 1:N_traj
        values[n, 1] = x0s[n]
        for i in 2:N_time+1
            dw = sqrt(dt) * randn()  # Wiener process increment
            values[n, i] = values[n, i-1] + process.θ * (-values[n, i-1]) * dt + process.σ * dw
        end
    end

    return times, values
end


using Random

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
    N_dim = length(x0)  # Dimension of x0
    N_time = round(Int, T/dt)  # Number of time steps
    times = 0:dt:T
    values = zeros(N_dim, N_time+1, K)
    
    for k in 1:K
        current_x = copy(x0)  # Copy initial condition for this trajectory
        for i in 2:N_time+1
            dw = sqrt(dt) * randn(N_dim)  # Multidimensional Wiener process increment
            current_x .= current_x .+ process.θ .* (-current_x) .* dt .+ process.σ .* dw
            values[:, i, k] = current_x
        end
    end

    return times, values
end


