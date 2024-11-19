using GrowthModels
using Test
using Plots
using SparseArrays
using ForwardDiff

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

    @test r_skiba.production_function_prime(0.1) == skiba_model.A_L * skiba_model.α * 0.1^(skiba_model.α - 1)

    @test isa(r_skiba, SolvedModel)
end

@testset "Non-Uniform Grid" begin
    skiba_model = SkibaModel()
    k_hps_non_unif = HyperParams(coef = 5.0, power = 10.0, N = 1000, xmax = 5.0, xmin = 1e-3);
    k_hps_unif = HyperParams(N = 1000, xmax = 5.0, xmin = 1e-3);

    k_hps_non_unif.power

    skiba_hyperparams_non_unif = StateSpaceHyperParams((k = k_hps_non_unif,));
    skiba_hyperparams_unif = StateSpaceHyperParams((k = k_hps_unif,));

    skiba_state_unif = StateSpace(skiba_model, skiba_hyperparams_unif)
    skiba_state_non_unif = StateSpace(skiba_model, skiba_hyperparams_non_unif)



    skiba_init_value_unif = Value(skiba_state_unif);
    skiba_init_value_non_unif = Value(skiba_state_non_unif);



    fit_value_unif, fit_variables_unif, fit_iter_unif = solve_HJB(
        skiba_model, 
        skiba_hyperparams_unif, 
        init_value = skiba_init_value_unif, maxit = 1000);
    
    fit_value_non_unif, fit_variables_non_unif, fit_iter_non_unif = solve_HJB(
        skiba_model, 
        skiba_hyperparams_non_unif, 
        init_value = skiba_init_value_non_unif, maxit = 1000);



    @test fit_value_non_unif.convergence_status == true
    @test fit_value_non_unif.v !== zeros(skiba_hyperparams_non_unif[:k].N)

    # make sure not changing in place
    # Graphs
    plot_diagnostics_output = plot_diagnostics(skiba_model, fit_value_non_unif, fit_variables_non_unif, skiba_hyperparams_non_unif)
    plot_model_output = plot_model(skiba_model, fit_value_non_unif, fit_variables_non_unif)

    @test isa(plot_model_output, Plots.Plot) 
    @test isa(plot_diagnostics_output, Plots.Plot) 
    # Model Output
    r_skiba = SolvedModel(skiba_model, fit_value_non_unif, fit_variables_non_unif)

    @test r_skiba.production_function_prime(0.1) == skiba_model.A_L * skiba_model.α * 0.1^(skiba_model.α - 1)

    @test isa(r_skiba, SolvedModel)

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


    # Evolution of capital goes to analytical steady state
    sm = SolvedModel(m, fit_value, fit_variables)
    g = fill(1, size(sm.value.A', 1)) 
    g = g ./ sum(g)
    distribution_time_series = StateEvolution(g, sm, 101)
    min_dist = minimum(abs.(vec(sm.variables[:k])[argmax(distribution_time_series[100])] .- k_steady_state(m)))
    # smooth skiba steady state not actually calculated analytically
    if !isa(m, SmoothSkibaModel)
        @test min_dist < 1e-2
    end

    # Model Output
    r = SolvedModel(m, fit_value, fit_variables)

    @test isa(r, SolvedModel)
end

model_names = ["StochasticRamseyCassKoopmansModel", "StochasticSkibaModel"]
@testset "Stochastic Model Tests for $model_name" for model_name in model_names
    # Dynamically instantiate the model based on its name
    m = eval(Meta.parse(model_name))()
    hyperparams = StateSpaceHyperParams(m, Nz = 40, Nk = 100, coef = 0, power = 0, kmax_f = 5.0)
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

    # FP
    sm = SolvedModel(m, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    grid_diag = GrowthModels.create_grid_diag(sm.variables[:k][:, 1], 40)
    g = fill(1, size(A_t, 1)) 
    g = g ./ sum(g)
    v_dim = size(init_value.v)
    
    distribution_time_series = StateEvolution(g, sm, 100)
    @test isa(distribution_time_series, StateEvolution)
end




@testset "Poisson Skiba" begin
    p = PoissonProcess(z = [-0.4, 0.0], λ = [9/10, 1/10])
    m = StochasticSkibaModel(p) 
    k_hps = HyperParams(N = 1000, xmax = 20, xmin = 1e-3)
    z_hps = HyperParams(N = 2, xmax = maximum(p.z), xmin = minimum(p.z))
    hyperparams = StateSpaceHyperParams((k = k_hps, z = z_hps))

    state = StateSpace(m, hyperparams)
    init_value = Value(state)
    v_dim = size(init_value.v)

    fit_value, fit_variables, fit_iter = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000)

    @test fit_value.convergence_status == true

    sm = SolvedModel(m, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    g = abs.(sin.(range(0, stop = 2π, length = size(A_t, 1))))
    g = g ./ sum(g)
    distribution_time_series =  StateEvolution(g, sm, 200);
end

@testset "Stochastic Ability Skiba" begin
    m = GrowthModels.StochasticSkibaAbilityModel()
    k_hps = HyperParams(coef = 5.0, power = 10.0, N = 1000, xmax = 20.0, xmin = 1e-3);
    z_hps = HyperParams(coef = 0.0, power = 0.0, N = 40, xmax = 1.5, xmin = 0.5);
    η_hps = HyperParams(coef = 0.0, power = 0.0, N = 5, xmax = 1.5, xmin = 0.5);
    hyperparams = StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
    state = StateSpace(m, hyperparams)
    value = Value(state);
    value.v[:] = GrowthModels.initial_guess(m, state)

    fit_value, fit_variables, fit_iter = solve_HJB(
        m, 
        hyperparams, 
        init_value = value, maxit = 1000);

    plot_model_output = plot_model(m, fit_value, fit_variables)

    r = SolvedModel(m, fit_value, fit_variables)

    sm = SolvedModel(m, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    g = abs.(sin.(range(0, stop = 2π, length = size(A_t, 1))))
    g = g ./ sum(g)
    distribution_time_series =  StateEvolution(g, sm, 5);
end

# Differentiation tests
include("test-differentiable.jl")
# Precision tests
include("test-precision.jl")
include("test-matlab-rbc-diffusion.jl")
