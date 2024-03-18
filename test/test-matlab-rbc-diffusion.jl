using GrowthModels
using Test
using CSV, Tables


@testset "Recover Matlab Estimates" begin
# Initial set-up using Ben Moll code

ga = 2 # CRRA utility with parameter gamma
rho = 0.05 # discount rate
alpha = 0.3 # CURVATURE OF PRODUCTION FUNCTION
d = 0.05 # DEPRECIATION RATE

# ORNSTEIN-UHLENBECK PROCESS parameters
Var = 0.07
zmean = exp(Var / 2) # MEAN OF LOG-NORMAL DISTRIBUTION N(0,Var)
Corr = 0.9
the = -log(Corr)
sig2 = 2 * the * Var

k_st = (alpha * zmean / (rho + d)) ^ (1 / (1 - alpha))

# Grid for capital
N = 10
kmin = 0.3 * k_st
kmax = 3 * k_st

# Grid for productivity
J = 4
zmin = zmean * 0.8
zmax = zmean * 1.2



# GrowthModels code
m = StochasticRamseyCassKoopmansModel(
    γ = 2.0,
    A = 1.0,
    ρ = 0.05,
    α = 0.3,
    δ = 0.05,
    stochasticprocess = from_stationary_OrnsteinUhlenbeckProcess(ρ = 0.9, stationary_σ = 0.07)
)
hyperparams = StateSpaceHyperParams(
    (k = HyperParams(N = 10, xmin = kmin, xmax = kmax),
    z = HyperParams(N = 4, xmin = zmin, xmax = zmax)
    )
)
state = StateSpace(m, hyperparams)
value = Value(state)



fit_value, fit_variables, fit_iter = solve_HJB(
    m, 
    hyperparams, 
    init_value = value, maxit = 1000);

ss =  fit_variables[:y] - m.δ .* fit_variables[:k] - fit_variables[:c]

data_dir = joinpath(@__DIR__, "test-data")
matlab_result =  Tables.matrix(
    CSV.File(
        joinpath(data_dir, "diffusion-rbc-matlab-output.csv"), header = false)
        )
max_diff = abs(maximum(ss .- matlab_result))

@test max_diff < 10^(-6)
end

# ga = 2 # CRRA utility with parameter gamma
# rho = 0.05 # discount rate
# alpha = 0.3 # CURVATURE OF PRODUCTION FUNCTION
# d = 0.05 # DEPRECIATION RATE

# # ORNSTEIN-UHLENBECK PROCESS parameters
# Var = 0.07
# zmean = exp(Var / 2) # MEAN OF LOG-NORMAL DISTRIBUTION N(0,Var)
# Corr = 0.9
# the = -log(Corr)
# sig2 = 2 * the * Var

# k_st = (alpha * zmean / (rho + d)) ^ (1 / (1 - alpha))

# # Grid for capital
# N = 10
# kmin = 0.3 * k_st
# kmax = 3 * k_st

# # Grid for productivity
# J = 4
# zmin = zmean * 0.8
# zmax = zmean * 1.2


# m = StochasticRamseyCassKoopmansModel(
#     γ = 2.0,
#     A = 1.0,
#     ρ = 0.05,
#     α = 0.3,
#     δ = 0.05,
#     stochasticprocess = from_stationary_OrnsteinUhlenbeckProcess(ρ = 0.9, stationary_σ = 0.07)
# )
# hyperparams = StateSpaceHyperParams(
#     (k = HyperParams(N = 1000, xmin = kmin, xmax = kmax),
#     z = HyperParams(N = 40, xmin = zmin, xmax = zmax)
#     )
# )
# state = StateSpace(m, hyperparams)
# value = Value(state)


# using BenchmarkTools
# @btime fit_value, fit_variables, fit_iter = solve_HJB(
#     m, 
#     hyperparams, 
#     init_value = value, maxit = 1000);
#     # 920.489 ms (13800 allocations: 792.80 MiB)


#     using ProfileView
#     @ProfileView.profview fit_value, fit_variables, fit_iter = solve_HJB(
#         m, 
#         hyperparams, 
#         init_value = value, maxit = 1000);
