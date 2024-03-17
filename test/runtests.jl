using GrowthModels
using Test
using Plots
    

    skiba_model = SkibaModel()
    skiba_hyperparams = StateSpaceHyperParams(skiba_model)
    skiba_state = StateSpace(skiba_model, skiba_hyperparams)
    skiba_init_value = Value(skiba_state);


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

@testset "RamseyCassKoopmans" begin
    m = RamseyCassKoopmansModel()
    hyperparams = StateSpaceHyperParams(m)
    state = StateSpace(m, hyperparams)
    init_value = Value(state);

    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2);
    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000);

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams[:k].N)
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
    m = SmoothSkibaModel(β = 10.0)
    hyperparams = StateSpaceHyperParams(m)
    state = StateSpace(m, hyperparams)
    init_value = Value(state);


    fail_fit_value, fail_fit_variables, fail_fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 2);
    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000);

    @test fit_value.convergence_status == true
    @test fail_fit_value.convergence_status == false
    @test fit_value.v !== zeros(hyperparams[:k].N)
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
include("test-matlab-rbc-diffusion.jl")