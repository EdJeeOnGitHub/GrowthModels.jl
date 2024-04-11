using GrowthModels
using Test
using Plots

using CUDA

CUDA.allowscalar(true)

skiba_model = SkibaModel()
skiba_hyperparams = StateSpaceHyperParams(skiba_model)
skiba_state = StateSpace(skiba_model, skiba_hyperparams)
skiba_init_value = Value(skiba_state);

gpu_skiba_model = SkibaModel{Float32}()
gpu_skiba_hyperparams = StateSpaceHyperParams(gpu_skiba_model)
gpu_skiba_state = StateSpace(gpu_skiba_model, gpu_skiba_hyperparams)
gpu_skiba_init_value = Value(gpu_skiba_state);



# @btime update_v(skiba_model, skiba_init_value, skiba_state, skiba_hyperparams);

# @btime gpu_update_v(gpu_skiba_model, gpu_skiba_init_value, gpu_skiba_state, gpu_skiba_hyperparams);


# function gpu_update_v(m::DeterministicModel, value::Value{T, N_v}, state::StateSpace, hyperparams::StateSpaceHyperParams; iter = 0, crit = 10^(-6), Delta = 1000, verbose = true) where {T, N_v}
# ##
m = gpu_skiba_model
value = gpu_skiba_init_value
state = gpu_skiba_state
hyperparams = gpu_skiba_hyperparams
Delta = Float32(1000)

value.v[:] = GrowthModels.initial_guess(m, state)
# m = skiba_model
# value = skiba_init_value
# state = skiba_state
# hyperparams = skiba_hyperparams
# Delta = 1000


# ##
    γ, ρ, δ = m.γ, m.ρ, m.δ
    (; v, dVf, dVb, dV0, dist) = value
    k, y = state[:k], state.aux_state[:y] # y isn't really a state but avoid computing it each iteration this way
    (; N, dx, xmax, xmin) = hyperparams[:k]
    dk, kmax, kmin = dx, xmax, xmin


    V = v
    # forward difference
    # dVf[1:N-1, 1] .= (V[2:N, 1] .- V[1:(N-1), 1])/dk
    # dVf[N, 1] = (y[N] - δ*kmax)^(-γ) # state constraint, for stability
    dVf[:, 1] .= vcat((V[2:N, 1] .- V[1:(N-1), 1])/dk, (maximum(y) - δ * kmax) ^ (-γ))

    # backward difference
    # dVb[2:N, 1] .= (V[2:N, 1] .- V[1:(N-1), 1])/dk
    # dVb[1, 1] = (y[1] - δ*kmin)^(-γ) # state constraint, for stability
    dVb[:, 1] .= vcat((minimum(y) - δ * kmin)^(-γ), (V[2:N, 1] .- V[1:(N-1), 1])/dk)

    # consumption and savings with forward difference
    cf = max.(dVf[:, 1], Float32(1e-3)).^(-1/γ)
    muf = y - δ .* k - cf
    Hf = (cf.^(1-γ))/(1-γ) + dVf[:, 1].*muf

    # consumption and savings with backward difference
    cb = max.(dVb[:, 1],Float32(1e-3)).^(-1/γ)
    mub = y - δ.*k - cb
    Hb = (cb.^(1-γ))/(1-γ) + dVb[:, 1].*mub

    # consumption and derivative of value function at steady state
    c0 = y - δ.*k
    dV0[:, 1] = max.(c0, Float32(1e-3)).^(-γ)
    H0 = (c0.^(1-γ))/(1-γ)

    # dV_upwind makes a choice of forward or backward differences based on
    # the sign of the drift    
    Ineither = (1 .- (muf .> 0)) .* (1 .- (mub .< 0))
    Iunique = (mub .< 0) .* (1 .- (muf .> 0)) + (1 .- (mub .< 0)) .* (muf .> 0)
    Iboth = (mub .< 0) .* (muf .> 0)
    Ib = Iunique .* (mub .< 0) + Iboth .* (Hb .>= Hf)
    If = Iunique .* (muf .> 0) + Iboth .* (Hf .>= Hb)
    I0 = Ineither

    # consumption
    c  = cf .* If + cb .* Ib + c0 .* I0
    u = (c.^(1-γ))/(1-γ)

    # CONSTRUCT MATRIX
    X = -Ib .* mub/dk
    Y = -If .* muf/dk + Ib .* mub/dk
    Z = If .* muf/dk
    # A = spdiagm(
    #     0 => Y,
    #     -1 => X[2:N],
    #     1 => Z[1:(N-1)]
    # );
    # sparse_I = sparse(I, N, N)

    Y_rows = CuArray(1:N);
    X_rows = CuArray(2:N);
    Z_rows = CuArray(1:(N-1));

    Y_cols = CuArray(1:N);
    X_cols = CuArray(1:(N-1));
    Z_cols = CuArray(2:N);

    rows = vcat(Y_rows, X_rows, Z_rows);
    cols = vcat(Y_cols, X_cols, Z_cols);
    vals = vcat(Y, X[2:N], Z[1:(N-1)]);
    A = CUSPARSE.CuSparseMatrixCSC(cols, rows, vals, (N, N));

    sparse_I = CUSPARSE.CuSparseMatrixCSC(Y_rows, Y_cols, CUDA.ones(Float32, N), (N, N));

    A_err = abs.(sum(A, dims = 2))        
    if maximum(A_err) > 10^(-6)
        throw(ValueFunctionError("Improper Transition Matrix: $(maximum(A_err)) > 10^(-6)"))
    end    

    B = (ρ + 1/Delta) * sparse_I .- A;
    # B = (ρ + 1/Delta) * sparse(I, N, N) .- A
    b = u + V/Delta
    cpu_B = Matrix(B)
    cpu_b = Vector(b)
    cpu_V = cpu_B\cpu_b
    V = B\b # SOLVE SYSTEM OF EQUATIONS
    Vchange = V - v
    # If using forward diff, want this just to be value part
    distance = ForwardDiff.value(maximum(abs.(Vchange)))
    push!(dist, distance)

    if distance < crit
        if verbose
            println("Value Function Converged, Iteration = ", iter)
        end
        push!(dist, distance)
        value = Value{T, N_v}(
            v = V, 
            dVf = dVf, 
            dVb = dVb, 
            dV0 = dV0, 
            dist = dist,
            convergence_status = true,
            iter = iter
            )
        variables = (
            y = y, 
            k = k, 
            c = c, 
            If = If, 
            Ib = Ib, 
            I0 = I0
            )
        return value, iter, variables
    end

    value = Value{T,N_v}(
        v = V, 
        dVf = dVf, 
        dVb = dVb, 
        dV0 = dV0, 
        dist = dist,
        convergence_status = false,
        iter = iter
        )

    return value, iter
end
using BenchmarkTools
@btime fit_value, fit_variables, fit_iter = solve_HJB(
    skiba_model, 
    skiba_hyperparams, 
    init_value = skiba_init_value, maxit = 1000);

@testset "Skiba" begin
    skiba_model = SkibaModel()
    skiba_hyperparams = StateSpaceHyperParams(skiba_model)
    skiba_state = StateSpace(skiba_model, skiba_hyperparams)
    skiba_init_value = Value(skiba_state);

    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(
        skiba_model, skiba_hyperparams, init_value = skiba_init_value, maxit = 2);
    fit_value, fit_variables, fit_iter = solve_HJB(
        skiba_model, 
        skiba_hyperparams, 
        init_value = skiba_init_value, maxit = 1000);

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(skiba_hyperparams[:k].N)
    @test isnothing(fail_fit_variables) == true 
    @test isnothing(fit_variables) == false 

    # make sure not changing in place
    old_value = deepcopy(fit_value)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(
        skiba_model, skiba_hyperparams, init_value = skiba_init_value, maxit = 2);
    @test old_value.v == fit_value.v
    @test old_value.dV0 == fit_value.dV0
    @test old_value.dVf == fit_value.dVf
    @test old_value.dVb == fit_value.dVb
    @test old_value.convergence_status == fit_value.convergence_status
    @test old_value.dist == fit_value.dist
    @test old_value.iter == fit_value.iter

    # Graphs
    plot_diagnostics_output = plot_diagnostics(skiba_model, fit_value, fit_variables, skiba_hyperparams)
    plot_model_output = plot_model(skiba_model, fit_value, fit_variables)

    @test isa(plot_model_output, Plots.Plot) 
    @test isa(plot_diagnostics_output, Plots.Plot) 
    # Model Output
    r_skiba = SolvedModel(skiba_model, fit_value, fit_variables)
    ode_skiba = r_skiba([0.1, 0.5, 1.0, 4.0], (0.0, 24.0))
    time_plot = plot_timepath(ode_skiba, r_skiba)


    @test r_skiba.production_function_prime(0.1) == skiba_model.A_L * skiba_model.α * 0.1^(skiba_model.α - 1)

    @test isa(r_skiba, SolvedModel)
    @test isa(time_plot, Plots.Plot)

end


model_names = ["RamseyCassKoopmansModel", "SkibaModel", "SmoothSkibaModel"]
@testset "Model Tests for $model_name" for model_name in model_names
    # Dynamically instantiate the model based on its name
    m = eval(Meta.parse(model_name))()
    hyperparams = StateSpaceHyperParams(m)
    state = StateSpace(m, hyperparams)
    init_value = Value(state)

    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2)
    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000)

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams[:k].N)
    @test isnothing(fail_fit_variables) == true
    @test isnothing(fit_variables) == false

    # Make sure not changing in place
    old_value = deepcopy(fit_value)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2)
    @test old_value.v == fit_value.v
    @test old_value.dV0 == fit_value.dV0
    @test old_value.dVf == fit_value.dVf
    @test old_value.dVb == fit_value.dVb
    @test old_value.convergence_status == fit_value.convergence_status
    @test old_value.dist == fit_value.dist
    @test old_value.iter == fit_value.iter

    # Graphs
    plot_diagnostics_output = plot_diagnostics(m, fit_value, fit_variables, hyperparams)
    plot_model_output = plot_model(m, fit_value, fit_variables)

    @test isa(plot_model_output, Plots.Plot)
    @test isa(plot_diagnostics_output, Plots.Plot)

    # Model Output
    r = SolvedModel(m, fit_value, fit_variables)
    ode = r([0.1, 0.5, 1.0, 4.0], (0.0, 24.0))
    time_plot = plot_timepath(ode, r)


    @test isa(r, SolvedModel)
    @test isa(time_plot, Plots.Plot)
end

model_names = ["StochasticRamseyCassKoopmansModel", "StochasticSkibaModel"]
@testset "Stochastic Model Tests for $model_name" for model_name in model_names
    # Dynamically instantiate the model based on its name
    m = eval(Meta.parse(model_name))()
    hyperparams = StateSpaceHyperParams(m, Nz = 40, Nk = 100)
    state = StateSpace(m, hyperparams)
    init_value = Value(state)

    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2)

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams[:k].N)
    @test isnothing(fail_fit_variables) == true
    @test isnothing(fit_variables) == false

    # Make sure not changing in place
    old_value = deepcopy(fit_value)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2)
    @test old_value.v == fit_value.v
    @test old_value.dV0 == fit_value.dV0
    @test old_value.dVf == fit_value.dVf
    @test old_value.dVb == fit_value.dVb
    @test old_value.convergence_status == fit_value.convergence_status
    @test old_value.dist == fit_value.dist
    @test old_value.iter == fit_value.iter

    # Graphs
    plot_diagnostics_output = plot_diagnostics(m, fit_value, fit_variables, hyperparams)
    plot_model_output = plot_model(m, fit_value, fit_variables)

    @test isa(plot_model_output, Plots.Plot)
    @test isa(plot_diagnostics_output, Plots.Plot)


    # TODO: add these for stochastic models or just remove

    # r = SolvedModel(m, fit_value, fit_variables)
    # ode = r([0.1, 0.5, 1.0, 4.0], (0.0, 24.0))
    # time_plot = plot_timepath(ode, r)


    # @test isa(r, SolvedModel)
    # @test isa(time_plot, Plots.Plot)
end


# Differentiation tests
include("test-differentiable.jl")
# Precision tests
include("test-precision.jl")
include("test-matlab-rbc-diffusion.jl")