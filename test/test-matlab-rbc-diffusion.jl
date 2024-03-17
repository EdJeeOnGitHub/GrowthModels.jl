# Clearing variables and figures is not typically done in Julia scripts
# using clear all; close all; clc; equivalent is not needed.
using GrowthModels
using BenchmarkTools
using LinearAlgebra, SparseArrays, Plots
using ForwardDiff



# function update_v(m::Union{StochasticSkibaModel}, value::Value{T, N_v}, state::StateSpace, hyperparams::StateSpaceHyperParams; iter = 0, crit = 10^(-6), Delta = 1000, verbose = true) where {T, N_v}

# Parameters
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
m = StochasticRamseyCassKoopmansModel(
    γ = 2.0,
    ρ = 0.05,
    α = 0.3,
    δ = 0.05,
    stochasticprocess = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ = 2*-log(0.9)*0.07 )
)

hyperparams = StateSpaceHyperParams(m, Nk = 10, Nz = 4)
state = StateSpace(m, hyperparams)
value = Value(state)
ed = similar(value.v)


initial_guess(m, state) = state.aux_state[:y] .^ (1 - m.γ) / (1 - m.γ) / m.ρ

v0 = initial_guess(m, state)
value.v .= initial_guess(m, state)

update_v(
    m,
    value,
    state,
    hyperparams
)

initial_guess(m, state)



    fit_value, fit_variables, fit_iter = solve_HJB(
        m, 
        hyperparams, 
        init_value = value, maxit = 1000);


fit_value



state[:k]

fit_variables[:y]
fit_variables[:c]
fit_variables[:k]

ss = state.aux_state[:y] - m.δ .* 

# Calculation before plotting
ss = zz .* kk.^alpha - d .* kk - c
using CSV, Tables
matlab_result =  Tables.matrix(CSV.File("test/test-data/diffusion-rbc-matlab-output.csv", header = false))

max_diff = abs(maximum(ss .- matlab_result))

@testset "Recover Matlab Estimates" begin
    @test max_diff < 10^(-6)
end


# # Creating the plot
# plot(k, ss, label="", linewidth=2, xlabel="k", ylabel="s(k,z)", legend=:topright)
# plot!(k, zeros(size(k)), linestyle=:dash, label="", linewidth=2)

# # Setting the font size for the plot is typically done through themes in Julia, or by adjusting default attributes
# plot!(tickfontsize=14, labelfontsize=14, legendfontsize=14, titlefontsize=14)

# # Setting x-axis limits
# xlims!(kmin, kmax)

# # Display the plot
# display(plot)










### Matlab code
using LinearAlgebra, SparseArrays, Plots

# Start measuring time
tic = time()

# Parameters
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
k = range(kmin, kmax, length = N)
dk = (kmax - kmin) / (N - 1)

# Grid for productivity
J = 4
zmin = zmean * 0.8
zmax = zmean * 1.2
z = range(zmin, zmax, length = J)
dz = (zmax - zmin) / (J - 1)
dz2 = dz ^ 2

kk = repeat(reshape(k, :, 1), 1, J)
zz = repeat(reshape(z, 1, :), N, 1)

# Drift and variance from Ito's lemma
mu = (-the * log.(z) .+ sig2 / 2) .* z
s2 = sig2 .* z .^ 2

maxit = 100
crit = 10 ^ (-6)
Delta = 1000

# Initialize matrices
Vaf = zeros(N, J)
Vab = zeros(N, J)
Vzf = zeros(N, J)
Vzb = zeros(N, J)
Vzz = zeros(N, J)
c = zeros(N, J)

# Constructing matrix Bswitch summarizing evolution of z
yy = -s2 / dz2 - mu / dz
chi = s2 / (2 * dz2)
zeta = mu / dz + s2 / (2 * dz2)


lowdiag = fill(chi[2], N)
for j in 3:J 
global    lowdiag = [lowdiag; fill(chi[j], N)]
end
lowdiag

updiag = Vector{Float64}()
for j in 1:(J - 1)
global    updiag = [updiag; fill(zeta[j], N)]
end
updiag


centdiag = fill(chi[1] + yy[1], N)
for j in 2:(J - 1)
global    centdiag = [centdiag; fill(yy[j], N)]
end
centdiag = [centdiag; fill(yy[end] + zeta[end], N)]
centdiag




# Construct B_switch matrix with corrected diagonals
Bswitch = spdiagm(-N => lowdiag, 0 => centdiag, N => updiag)

# Initial guess
v0 = (zz .* kk .^ alpha) .^ (1 - ga) / (1 - ga) / rho
global v = v0

# for n = 1:maxit
n = 1
global    V = v
    # Forward difference
    Vaf[1:N-1, :] = (V[2:N, :] - V[1:N-1, :]) ./ dk

    # Backward difference
    Vab[2:N, :] = (V[2:N, :] - V[1:N-1, :]) ./ dk
    Vab[1, :] .= (z .* kmin.^alpha .- d .* kmin).^(-ga) # State constraint at kmin

    # Indicator whether value function is concave
    I_concave = Vab .> Vaf

    # Consumption and savings with forward difference
    cf = Vaf.^(-1 / ga)
    sf = zz .* kk.^alpha - d .* kk - cf
    # Consumption and savings with backward difference
    cb = Vab.^(-1 / ga)
    sb = zz .* kk.^alpha - d .* kk - cb
    # Consumption and derivative of value function at steady state
    c0 = zz .* kk.^alpha - d .* kk
    Va0 = c0.^(-ga)

    # Decision on forward or backward differences based on the sign of the drift
    If = sf .> 0 # positive drift -> forward difference
    Ib = sb .< 0 # negative drift -> backward difference
    I0 = 1 .- If .- Ib # at steady state

    Va_Upwind = Vaf .* If + Vab .* Ib + Va0 .* I0

    c .= Va_Upwind.^(-1 / ga)
    u = c.^(1 - ga) / (1 - ga)

    # Construct matrix A
    X = -min.(sb, 0) ./ dk
    Y = -max.(sf, 0) ./ dk + min.(sb, 0) ./ dk
    Z = max.(sf, 0) ./ dk





    # Initialize updiag with zeros, to be filled in the loop
    updiag = Vector{Float64}()  # Start with an empty array, assuming Z contains Float64 values
    for j in 1:J
        updiag = [updiag; Z[1:N-1, j]]
        if j != J
            push!(updiag, 0)
        end
    end
    updiag



    # Convert centdiag
    centdiag = reshape(Y, N*J)  # Assuming Y is already a matrix or array that matches the dimensions

    lowdiag = X[2:end, 1]
    for j in 2:J
        lowdiag = [lowdiag; 0; X[2:end, j]]
    end
    lowdiag


    AA = spdiagm(0 => centdiag, 1 => updiag, -1 => lowdiag)


    A = AA + Bswitch

    if maximum(abs.(sum(A, dims=2))) > 10^(-12)
        println("Improper Transition Matrix")
        break
    end

    B = (1 / Delta + rho) * sparse(I, size(A)) .- A



    u_stacked = reshape(u, N*J)
    V_stacked = reshape(V, N*J)

    b = u_stacked + V_stacked / Delta

    V_stacked = B \ b

    V = reshape(V_stacked, N, J)

    Vchange = V - v
global    v = V

    dist = maximum(abs.(Vchange))
    if dist < crit
        println("Value Function Converged, Iteration = ", n)
        break
    end
end
dist
# Measure elapsed time
toc = time() - tic
println("Elapsed time: $toc seconds")


# Calculation before plotting
ss = zz .* kk.^alpha - d .* kk - c
using CSV, Tables
matlab_result =  Tables.matrix(CSV.File("test/test-data/diffusion-rbc-matlab-output.csv", header = false))

max_diff = abs(maximum(ss .- matlab_result))

@testset "Recover Matlab Estimates" begin
    @test max_diff < 10^(-6)
end


# # Creating the plot
# plot(k, ss, label="", linewidth=2, xlabel="k", ylabel="s(k,z)", legend=:topright)
# plot!(k, zeros(size(k)), linestyle=:dash, label="", linewidth=2)

# # Setting the font size for the plot is typically done through themes in Julia, or by adjusting default attributes
# plot!(tickfontsize=14, labelfontsize=14, legendfontsize=14, titlefontsize=14)

# # Setting x-axis limits
# xlims!(kmin, kmax)

# # Display the plot
# display(plot)

