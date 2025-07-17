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
    g = fill(1.0, size(sm.value.A', 1)) 

    dx_stacked = create_dx_stacked(sm.variables[:k][:, 1], size(sm.variables.k, 2))

    T_max = 100
    distribution_time_series = StateEvolution(g, sm, T_max)
    cell_masses = distribution_time_series.cell_masses
    mass_over_t = dropdims(sum(cell_masses, dims = 1), dims = 1)
    for i in 1:T_max
        @test isapprox(mass_over_t[i], 1.0; atol = 1e-2)
    end

    # test that evolution converges to analytical steady state
    E_k = cell_masses .* sm.variables.k
    min_dist = minimum(abs.(vec(sm.variables[:k])[argmax(E_k[:, end])] .- k_steady_state(m)))
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
    plot_diagnostics_output = plot_diagnostics(m, fit_value, fit_variables, hyperparams);
    plot_model_output = plot_model(m, fit_value, fit_variables);

    @test isa(plot_model_output, Plots.Plot)
    @test isa(plot_diagnostics_output, Plots.Plot)

    # FP
    sm = SolvedModel(m, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    # test colsums of A close to 0
    for i in 1:size(A_t, 2)
        @test isapprox(sum(A_t[:, i]), 0.0; atol = 1e-10)
    end
    g = fill(1.0, size(A_t, 1)) 
    dx_stacked = GrowthModels.create_dx_stacked(sm.variables[:k][:, 1], size(sm.variables.k, 2))
    
    distribution_time_series = StateEvolution(g, sm, 10);
    cell_mass = distribution_time_series.cell_masses
    cell_mass_by_t = dropdims(sum(cell_mass, dims = (1, 2)), dims = (1, 2))
    for t in 1:size(cell_mass_by_t, 1)
        println("Weighted mass at time $t: ", cell_mass_by_t[t])
        @test isapprox(cell_mass_by_t[t], 1.0; atol = 1e-8)
    end
end


@testset "Poisson Skiba" begin
    Q = [-9/10 9/10; 1/10 -1/10]
    p = PoissonProcess(z = [-0.4, 0.0], Q = Q)
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

    distribution_time_series =  StateEvolution(g, sm, 200; implicit_steps = 1);

    cell_masses = distribution_time_series.cell_masses
    cell_masses_by_t = dropdims(sum(cell_masses, dims = (1, 2)), dims = (1, 2))
    for t in 1:size(cell_masses_by_t, 1)
        println("Weighted mass at time $t: ", cell_masses_by_t[t])
        @test isapprox(cell_masses_by_t[t], 1.0; atol = 1e-8)
    end

end


# Check that transition matrices never try and move people across abilities
function index_checker(grid_dims, A, B; idx_to_check = 1)
    (Nk, Nz, Nη) = grid_dims
    # this gives a vector of length Nk * Nz * Nη
    # where we can give it a number in Nk*Nz*Nη and it will return the 
    # corresponding CartesianIndex
    cart_idxs = CartesianIndices((Nk, Nz, Nη))

    A_nz = findnz(A[idx_to_check, :])[1]
    B_nz = findnz(B[idx_to_check, :])[1]

    A_carts = cart_idxs[A_nz]
    B_carts = cart_idxs[B_nz]

    A_abilities = [x[3] for x in A_carts]
    B_abilities = [x[3] for x in B_carts]

    A_check = all(cart_idxs[idx_to_check][3] .== A_abilities)
    B_check = all(cart_idxs[idx_to_check][3] .== B_abilities)
    return A_check, B_check
end

function setup_ability_check(sm, g)
    dx_stacked = create_dx_stacked(sm.variables[:k][:, 1, 1], prod(size(sm.variables.k)[2:end]))
    normalize_by_weighted_mass!(g, dx_stacked)

    init_mass =  g .* dx_stacked
    init_mass_grid = reshape(init_mass, size(sm.variables[:k]))
    init_ability = dropdims(sum(init_mass_grid, dims = (1, 2)), dims = (1, 2))

    evolution = StateEvolution(g, sm, 5; implicit_steps = 1)
    cell_masses = evolution.cell_masses
    ability_by_t = dropdims(sum(cell_masses, dims = (1, 2)), dims = (1, 2))
    return init_ability, ability_by_t, evolution, dx_stacked
end

@testset "Stochastic Ability Skiba" begin
    Nk = 1000
    Nz = 40
    Nη = 5

    m_a = GrowthModels.StochasticSkibaAbilityModel()
    k_hps = HyperParams(coef = 5.0, power = 10.0, N = Nk, xmax = 20.0, xmin = 1e-3);
    z_hps = HyperParams(coef = 0.0, power = 0.0, N = Nz, xmax = 1.5, xmin = 0.5);
    η_hps = HyperParams(coef = 0.0, power = 0.0, N = Nη, xmax = 1.5, xmin = 0.5);
    hyperparams_a = StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
    state_a = StateSpace(m_a, hyperparams_a)
    value_a = Value(state_a);
    value_a.v[:] = GrowthModels.initial_guess(m_a, state_a)
    fit_value, fit_variables, fit_iter = solve_HJB(
        m_a, 
        hyperparams_a, 
        init_value = value_a, maxit = 1000);

    Bswitch_a = GrowthModels.construct_diffusion_matrix(m_a.stochasticprocess, state_a, hyperparams_a)
    @test all(isapprox.(sum(Bswitch_a, dims = 2), zeros(size(Bswitch_a, 1)); atol = 1e-10)  .== 1)


    plot_model_output = plot_model(m_a, fit_value, fit_variables)


    grid_dims = (Nk, Nz, Nη)
    is_to_check = collect(1:prod(grid_dims))
    is_to_check = sample(is_to_check, 1000)
    idx_checks = []
    for idx_to_check in is_to_check
        id_c = index_checker(grid_dims, fit_value.A, Bswitch_a; idx_to_check = idx_to_check)
        push!(idx_checks, id_c)
    end

    for idx_check in idx_checks
        @test idx_check[1] == true
        @test idx_check[2] == true
    end



    r = SolvedModel(m_a, fit_value, fit_variables)
    sm = SolvedModel(m_a, fit_value, fit_variables)

    A_t = sparse(sm.value.A')
    # test colsums of A close to 0
    for i in 1:size(A_t, 2)
        @test isapprox(sum(A_t[:, i]), 0.0; atol = 1e-10)
    end


    g_init = abs.(sin.(range(0, stop = 2π, length = size(r.value.A', 1))))
    g_init = g_init ./ sum(g_init)
    init_ability, ability_by_t, evolution, dx_stacked = setup_ability_check(sm, g_init)
    # This isn't particularly precise - ability seems to be drifting
    for t in 1:size(ability_by_t, 2)
        comp_ab = hcat(ability_by_t[:, t], init_ability)
        println("Ability at time $t vs init ability: ", comp_ab)
        @test isapprox(ability_by_t[:, t], init_ability; rtol = 1e-1)
    end
end



@testset "Stochastic NP Ability" begin
    np_model = StochasticNPAbilityModel()

    np_hyperparams = StateSpaceHyperParams(np_model)
    np_state = StateSpace(np_model, np_hyperparams)
    np_init_value = Value(np_state)

    fit_value, fit_variables, fit_iter = solve_HJB(np_model, np_hyperparams, init_value = np_init_value, maxit = 1000)

    np_sm = SolvedModel(np_model, fit_value, fit_variables)

    Bswitch = GrowthModels.construct_diffusion_matrix(np_model.stochasticprocess, np_state, np_hyperparams)

    A_t = sparse(np_sm.value.A')
    # test colsums of A close to 0
    for i in 1:size(A_t, 2)
        @test isapprox(sum(A_t[:, i]), 0.0; atol = 1e-10)
    end
    g = abs.(sin.(range(0, stop = 2π, length = size(A_t, 1))))
    g = g ./ sum(g)


    grid_dims = size(np_sm.value.v)
    is_to_check = collect(1:prod(grid_dims))
    is_to_check = sample(is_to_check, 1000)
    idx_checks = []

    for idx_to_check in is_to_check
        id_c = index_checker(grid_dims, fit_value.A, Bswitch; idx_to_check = idx_to_check)
        push!(idx_checks, id_c)
    end

    for idx_check in idx_checks
        @test idx_check[1] == true
        @test idx_check[2] == true
    end

    g_init = g
    init_ability, ability_by_t, distribution_time_series, dx_stacked = setup_ability_check(np_sm, g_init)
    # This isn't particularly precise - ability seems to be drifting
    for t in 1:size(ability_by_t, 2)
        comp_ab = hcat(ability_by_t[:, t], init_ability)
        println("Ability at time $t vs init ability: ", comp_ab)        
        @test isapprox(ability_by_t[:, t], init_ability; rtol = 1e-1)   
    end
end



@testset "Stochastic NP Ability - Poisson" begin
    z = [0.4, 1.0, 3.5]
    Q = [-9/10 8/10 1/10; 1/10 -2/10 1/10; 1/10 1/10 -2/10]
    p = PoissonProcess(z = z, Q = Q)

    np_model = StochasticNPAbilityModel(p)

    np_hyperparams = StateSpaceHyperParams(np_model)
    np_state = StateSpace(np_model, np_hyperparams)
    np_init_value = Value(np_state)

    fit_value, fit_variables, fit_iter = solve_HJB(np_model, np_hyperparams, init_value = np_init_value, maxit = 1000)

    np_sm = SolvedModel(np_model, fit_value, fit_variables)

    Bswitch = GrowthModels.construct_diffusion_matrix(np_model.stochasticprocess, np_state, np_hyperparams)


    A_t = sparse(np_sm.value.A')
    # test colsums of A close to 0
    for i in 1:size(A_t, 2)
        @test isapprox(sum(A_t[:, i]), 0.0; atol = 1e-10)
    end
    g = abs.(sin.(range(0, stop = 2π, length = size(A_t, 1))))
    g = g ./ sum(g)

    grid_dims = size(np_sm.value.v)
    is_to_check = collect(1:prod(grid_dims))
    is_to_check = sample(is_to_check, 1000)
    idx_checks = []

    for idx_to_check in is_to_check
        id_c = index_checker(grid_dims, fit_value.A, Bswitch; idx_to_check = idx_to_check)
        push!(idx_checks, id_c)
    end

    for idx_check in idx_checks
        @test idx_check[1] == true
        @test idx_check[2] == true
    end

    g_init = g
    init_ability, ability_by_t, distribution_time_series, dx_stacked = setup_ability_check(np_sm, g_init)
    # This isn't particularly precise - ability seems to be drifting
    for t in 1:size(ability_by_t, 2)
        comp_ab = hcat(ability_by_t[:, t], init_ability)
        println("Ability at time $t vs init ability: ", comp_ab)    
        @test isapprox(ability_by_t[:, t], init_ability; rtol = 1e-1)       
    end
end


# Differentiation tests
include("test-differentiable.jl")
# Precision tests
include("test-precision.jl")
include("test-matlab-rbc-diffusion.jl")
include("test-mass-conservation.jl")