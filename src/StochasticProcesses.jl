
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
