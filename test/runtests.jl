using GrowthModels
using Test
using Plots

@testset "Skiba" begin
    m_skiba = SkibaModel()
    hyperparams = HyperParams(m_skiba)
    init_value = Value(hyperparams);


    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 2);
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams.N)
    @test isnothing(fail_fit_variables) == true 
    @test isnothing(fit_variables) == false 

    # make sure not changing in place
    old_value = deepcopy(fit_value)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 2);
    @test old_value.v == fit_value.v
    @test old_value.dV0 == fit_value.dV0
    @test old_value.dVf == fit_value.dVf
    @test old_value.dVb == fit_value.dVb
    @test old_value.convergence_status == fit_value.convergence_status
    @test old_value.dist == fit_value.dist
    @test old_value.iter == fit_value.iter

    # Graphs
    plot_diagnostics_output = plot_diagnostics(m_skiba, fit_value, fit_variables, hyperparams)
    plot_model_output = plot_model(m_skiba, fit_value, fit_variables)

    @test isa(plot_model_output, Plots.Plot) 
    @test isa(plot_diagnostics_output, Plots.Plot) 

    # Model Output
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    ode_skiba = r_skiba([0.1, 0.5, 1.0, 4.0], (0.0, 24.0))
    time_plot = plot_timepath(ode_skiba, r_skiba)


    @test r_skiba.production_function_prime(0.1) == m_skiba.A_L * m_skiba.α * 0.1^(m_skiba.α - 1)

    @test isa(r_skiba, SolvedModel)
    @test isa(time_plot, Plots.Plot)

end

@testset "RamseyCassKoopmans" begin
    m = RamseyCassKoopmansModel()
    hyperparams = HyperParams(m)
    init_value = Value(hyperparams);


    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2);
    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000);

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams.N)
    @test isnothing(fail_fit_variables) == true 
    @test isnothing(fit_variables) == false 

    # make sure not changing in place
    old_value = deepcopy(fit_value)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2);
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


    @test r.production_function_prime(0.1) == m.A * m.α * 0.1^(m.α - 1)

    @test isa(r, SolvedModel)
    @test isa(time_plot, Plots.Plot)

end


@testset "SmoothSkiba" begin
    m = SmoothSkibaModel(β = 100.0, δ = 0.05, κ = 2.0, ρ = 0.2)
    hyperparams = HyperParams(m)
    init_value = Value(hyperparams);


    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2);
    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000);

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams.N)
    @test isnothing(fail_fit_variables) == true 
    @test isnothing(fit_variables) == false 

    # make sure not changing in place
    old_value = deepcopy(fit_value)
    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2);
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

    plot_k = collect(fit_variables.k[10:10:end])
    # Model Output
    r = SolvedModel(m, fit_value, fit_variables)
    ode = r(plot_k, (0.0, 48.0))
    plot_model_output
    time_plot = plot_timepath(ode, r)

    @test isa(r, SolvedModel)
    @test isa(time_plot, Plots.Plot)

end


# Differentiation tests
include("test-differentiable.jl")
# Precision tests
include("test-precision.jl")