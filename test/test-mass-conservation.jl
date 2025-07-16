using GrowthModels
using Test
using SparseArrays
using LinearAlgebra

"""
Mass Conservation Tests for StateEvolution.jl

This test suite addresses the 5 critical issues identified in mass-conservation-guide.md:
1. Raw g summing vs weighted mass
2. Incorrect identity matrix construction  
3. Mis-weighted mass matrix in weak form
4. Non-conservative boundary conditions
5. Incorrect time-step in times overload
"""

# Helper function to create a simple test setup
function create_test_setup(; uniform_grid = true, N = 100)
    # Create a simple model for testing
    model = SkibaModel()
    
    if uniform_grid
        k_hps = HyperParams(N = N, xmax = 5.0, xmin = 1e-3, coef = 0.0, power = 0.0)
    else
        # Non-uniform grid with power law spacing
        k_hps = HyperParams(N = N, xmax = 5.0, xmin = 1e-3, coef = 5.0, power = 10.0)
    end
    
    hyperparams = StateSpaceHyperParams((k = k_hps,))
    state = StateSpace(model, hyperparams)
    init_value = Value(state)
    
    # Solve HJB to get transition matrix
    fit_value, fit_variables, _ = solve_HJB(model, hyperparams, init_value = init_value, maxit = 1000)
    
    # Create solved model
    sm = SolvedModel(model, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    
    # Create grid spacing
    k = sm.variables.k
    dkf, dkb = GrowthModels.generate_dx(k)
    dx_tilde = 0.5 * (dkf + dkb)
    dx_tilde[1] = 0.5 * dkf[1]
    dx_tilde[end] = 0.5 * dkb[end]
    
    grid_diag = spdiagm(0 => dx_tilde)
    
    return A_t, grid_diag, dx_tilde, sm
end

# Helper function to create initial distribution
function create_initial_distribution(N; normalized = true)
    g = abs.(sin.(range(0, stop = 2π, length = N)))
    if normalized
        g = g ./ sum(g)
    end
    return g
end

# Helper function to create 2D multivariate test setup (k, z dimensions)
function create_2d_test_setup(; uniform_grid = true, Nk = 100, Nz = 40)
    # Create Ornstein-Uhlenbeck process for z dimension
    ou_process = OrnsteinUhlenbeckProcess(θ = -log(0.9), σ = 0.1)
    model = StochasticSkibaModel(ou_process)
    
    if uniform_grid
        k_hps = HyperParams(N = Nk, xmax = 5.0, xmin = 1e-3, coef = 0.0, power = 0.0)
    else
        # Non-uniform grid with power law spacing
        k_hps = HyperParams(N = Nk, xmax = 5.0, xmin = 1e-3, coef = 5.0, power = 10.0)
    end
    
    # Create z dimension hyperparams
    zmean = GrowthModels.process_mean(ou_process)
    zmax = zmean * 1.2
    zmin = zmean * 0.8
    z_hps = HyperParams(N = Nz, xmax = zmax, xmin = zmin)
    
    hyperparams = StateSpaceHyperParams((k = k_hps, z = z_hps))
    state = StateSpace(model, hyperparams)
    init_value = Value(state)
    
    # Solve HJB to get transition matrix
    fit_value, fit_variables, _ = solve_HJB(model, hyperparams, init_value = init_value, maxit = 1000)
    
    # Create solved model
    sm = SolvedModel(model, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    
    # Create grid spacing for 2D case
    k = sm.variables.k[:, 1]  # Extract k dimension
    grid_diag = create_grid_diag(k, Nz)
    
    return A_t, grid_diag, sm, (Nk, Nz)
end

# Helper function to create 2D test setup with Poisson process
function create_2d_poisson_test_setup(; uniform_grid = true, Nk = 100)
    # Create Poisson process for z dimension
    Q = [-9/10 9/10; 1/10 -1/10]
    p = PoissonProcess(z = [-0.4, 0.0], Q = Q)
    model = StochasticSkibaModel(p)
    
    if uniform_grid
        k_hps = HyperParams(N = Nk, xmax = 5.0, xmin = 1e-3, coef = 0.0, power = 0.0)
    else
        # Non-uniform grid with power law spacing
        k_hps = HyperParams(N = Nk, xmax = 5.0, xmin = 1e-3, coef = 5.0, power = 10.0)
    end
    
    # Create z dimension hyperparams (2 discrete states)
    Nz = 2
    z_hps = HyperParams(N = Nz, xmax = maximum(p.z), xmin = minimum(p.z))
    
    hyperparams = StateSpaceHyperParams((k = k_hps, z = z_hps))
    state = StateSpace(model, hyperparams)
    init_value = Value(state)
    
    # Solve HJB to get transition matrix
    fit_value, fit_variables, _ = solve_HJB(model, hyperparams, init_value = init_value, maxit = 1000)
    
    # Create solved model
    sm = SolvedModel(model, fit_value, fit_variables)
    A_t = sparse(sm.value.A')
    
    # Create grid spacing for 2D case
    k = sm.variables.k[:, 1]  # Extract k dimension
    grid_diag = create_grid_diag(k, Nz)
    
    return A_t, grid_diag, sm, (Nk, Nz)
end

@testset "Mass Conservation Tests" begin
    @testset "Issue #1: Raw g summing vs weighted mass" begin
        @testset "Uniform Grid" begin
            A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = true, N = 100)
            g_init = create_initial_distribution(size(A_t, 1), normalized = false)
            
            # Normalize by weighted mass (correct approach)
            initial_weighted_mass = sum(g_init .* dx_tilde)
            g_init_normalized = g_init ./ initial_weighted_mass
            
            raw_mass = sum(g_init_normalized)
            weighted_mass = sum(g_init_normalized .* dx_tilde)
            
            @test weighted_mass ≈ 1.0 atol = 1e-10
            # On uniform grid, raw sum and weighted sum should be similar (proportional)
            @test abs(weighted_mass - raw_mass * dx_tilde[2]) < 1e-10  # Should be proportional
            
            # Test evolution preserves weighted mass
            g_evolved = GrowthModels.iterate_g(g_init_normalized, A_t, grid_diag, time_step = 0.1)
            evolved_weighted_mass = sum(g_evolved .* dx_tilde)
            
            @test evolved_weighted_mass ≈ weighted_mass atol = 1e-10
            
            println("Uniform grid - Initial weighted mass: ", weighted_mass)
            println("Uniform grid - Evolved weighted mass: ", evolved_weighted_mass)
            println("Uniform grid - dx_tilde[2]: ", dx_tilde[2])
            println("Uniform grid - Raw mass: ", raw_mass)

        end
        
        @testset "Non-uniform Grid" begin
            A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = false, N = 100)
            g_init = create_initial_distribution(size(A_t, 1), normalized = false)
            
            # Normalize by weighted mass (correct approach)
            initial_weighted_mass = sum(g_init .* dx_tilde)
            g_init_normalized = g_init ./ initial_weighted_mass
            
            raw_mass = sum(g_init_normalized)
            weighted_mass = sum(g_init_normalized .* dx_tilde)
            
            @test weighted_mass ≈ 1.0 atol = 1e-10
            # On non-uniform grid, raw sum and weighted sum should differ significantly
            @test abs(raw_mass - weighted_mass) > 1e-2
            
            # Test evolution preserves weighted mass
            g_evolved = GrowthModels.iterate_g(g_init_normalized, A_t, grid_diag, time_step = 0.1)
            evolved_weighted_mass = sum(g_evolved .* dx_tilde)
            
            @test evolved_weighted_mass ≈ 1.0 atol = 1e-10
            
            println("Non-uniform grid - Initial raw mass: ", raw_mass)
            println("Non-uniform grid - Initial weighted mass: ", weighted_mass)
            println("Non-uniform grid - Evolved weighted mass: ", evolved_weighted_mass)
        end
        
        @testset "Multiple Time Steps" begin
            A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = false, N = 100)
            g = create_initial_distribution(size(A_t, 1), normalized = false)
            
            # Normalize by weighted mass
            initial_weighted_mass = sum(g .* dx_tilde)
            g = g ./ initial_weighted_mass
            
            # Test that weighted mass is preserved over multiple time steps
            weighted_masses = Float64[]
            push!(weighted_masses, sum(g .* dx_tilde))
            
            for i in 1:10
                g = GrowthModels.iterate_g(g, A_t, grid_diag, time_step = 0.1)
                push!(weighted_masses, sum(g .* dx_tilde))
            end
            
            # All weighted masses should be approximately 1.0
            for mass in weighted_masses
                @test mass ≈ 1.0 atol = 1e-10
            end
            
            println("Weighted masses over time: ", weighted_masses)
        end
    end
    
    @testset "Demonstration of Issue" begin
        # This test demonstrates why raw summing fails on non-uniform grids
        A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = false, N = 100)
        g = create_initial_distribution(size(A_t, 1), normalized = false)
        
        # Normalize by RAW sum (incorrect approach)
        g_raw_normalized = g ./ sum(g)
        normalize_by_weighted_mass!(g_raw_normalized, grid_diag)
        
        # Test evolution - this should show mass drift
        raw_masses = Float64[]
        weighted_masses = Float64[]
        
        g_current = copy(g_raw_normalized)
        for i in 1:10
            push!(raw_masses, sum(g_current))
            push!(weighted_masses, sum(grid_diag .* g_current))
            g_current = GrowthModels.iterate_g(g_current, A_t, grid_diag, time_step = 0.1)
        end
        
        println("Raw masses (should drift): ", raw_masses)
        println("Weighted masses (should stay constant): ", weighted_masses)
        
        # Raw masses should drift significantly
        @test abs(raw_masses[end] - raw_masses[1]) > 1e-3
        
        # Weighted masses should stay relatively constant (though not perfect due to other issues)
        @test abs(weighted_masses[end] - weighted_masses[1]) < abs(raw_masses[end] - raw_masses[1])
    end
    
    @testset "New Utility Functions" begin
        A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = false, N = 100)
        g = create_initial_distribution(size(A_t, 1), normalized = false)
        
        @testset "weighted_mass function" begin
            # Test weighted_mass function
            manual_weighted_mass = sum(g .* dx_tilde)
            utility_weighted_mass = weighted_mass(g, grid_diag)

            
            @test manual_weighted_mass ≈ utility_weighted_mass atol = 1e-14
            println("Manual weighted mass: ", manual_weighted_mass)
            println("Utility weighted mass: ", utility_weighted_mass)
        end
        
        @testset "normalize_by_weighted_mass function" begin
            # Test normalize_by_weighted_mass (non-mutating)
            g_normalized = normalize_by_weighted_mass(g, grid_diag)
            normalized_mass = weighted_mass(g_normalized, grid_diag)
            
            @test normalized_mass ≈ 1.0 atol = 1e-14
            @test g != g_normalized  # Original should be unchanged
            
            # Test normalize_by_weighted_mass! (mutating)
            g_copy = copy(g)
            normalize_by_weighted_mass!(g_copy, grid_diag)
            mutated_mass = weighted_mass(g_copy, grid_diag)
            
            @test mutated_mass ≈ 1.0 atol = 1e-14
            @test g_copy ≈ g_normalized atol = 1e-14
            
            println("Normalized mass (non-mutating): ", normalized_mass)
            println("Normalized mass (mutating): ", mutated_mass)
        end
    end
    
    # @testset "StationaryDistribution Fix" begin
    #     A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = false, N = 100)
        
    #     # Test that StationaryDistribution now produces properly normalized distributions
    #     stationary_dist = StationaryDistribution(A_t, grid_diag)
    #     stationary_mass = weighted_mass(stationary_dist, grid_diag)
        
    #     @test stationary_mass ≈ 1.0 atol = 1e-10
    #     @test all(stationary_dist .>= 0)  # All probabilities should be non-negative
        
    #     println("Stationary distribution weighted mass: ", stationary_mass)
    #     println("Stationary distribution min/max: ", minimum(stationary_dist), " / ", maximum(stationary_dist))
    # end

    
    @testset "StateEvolution with SolvedModel" begin
        # Test that the high-level StateEvolution interface works correctly
        A_t, grid_diag, dx_tilde, sm = create_test_setup(uniform_grid = false, N = 100)
        g = create_initial_distribution(size(A_t, 1), normalized = false)
        
        # Normalize properly
        g_normalized = normalize_by_weighted_mass(g, grid_diag)
        
        # Test StateEvolution constructor that takes SolvedModel
        state_evolution = StateEvolution(g_normalized, sm, 5)
        
        @test isa(state_evolution, StateEvolution)
        @test size(state_evolution.S, 2) == 5  # 0, 1, 2, 3, 4, 5
        
        # Check that mass is conserved throughout evolution
        for i in 1:size(state_evolution.S, 2)
            mass_at_time_i = weighted_mass(state_evolution.S[:, i], grid_diag)
            println("Mass at time ", i, ": ", mass_at_time_i)
            @test mass_at_time_i ≈ 1.0 atol = 1e-10
        end
        
        println("StateEvolution mass conservation check passed")
    end
    
    @testset "Multivariate State Mass Conservation" begin
        @testset "2D States (k, z) with Ornstein-Uhlenbeck Process" begin
            @testset "Uniform Grid" begin
                A_t, grid_diag, sm, (Nk, Nz) = create_2d_test_setup(uniform_grid = true, Nk = 50, Nz = 20)
                g_init = create_initial_distribution(size(A_t, 1), normalized = false)
                
                # Normalize by weighted mass (correct approach)
                g_init_normalized = normalize_by_weighted_mass(g_init, grid_diag)
                
                # Check initial mass
                initial_weighted_mass = weighted_mass(g_init_normalized, grid_diag)
                @test initial_weighted_mass ≈ 1.0 atol = 1e-10
                
                # Test evolution preserves weighted mass
                g_evolved = GrowthModels.iterate_g(g_init_normalized, A_t, grid_diag, time_step = 0.1)
                evolved_weighted_mass = weighted_mass(g_evolved, grid_diag)
                
                @test evolved_weighted_mass ≈ 1.0 atol = 1e-10
                
                println("2D Uniform grid (k,z) - Initial weighted mass: ", initial_weighted_mass)
                println("2D Uniform grid (k,z) - Evolved weighted mass: ", evolved_weighted_mass)
                println("2D Grid dimensions: ", (Nk, Nz), " Total size: ", size(A_t, 1))
            end
            
            @testset "Non-uniform Grid" begin
                A_t, grid_diag, sm, (Nk, Nz) = create_2d_test_setup(uniform_grid = false, Nk = 50, Nz = 20)
                g_init = create_initial_distribution(size(A_t, 1), normalized = false)
                
                # Normalize by weighted mass (correct approach)
                g_init_normalized = normalize_by_weighted_mass(g_init, grid_diag)
                
                # Check initial mass
                initial_weighted_mass = weighted_mass(g_init_normalized, grid_diag)
                @test initial_weighted_mass ≈ 1.0 atol = 1e-10
                
                # Test evolution preserves weighted mass
                g_evolved = GrowthModels.iterate_g(g_init_normalized, A_t, grid_diag, time_step = 0.1)
                evolved_weighted_mass = weighted_mass(g_evolved, grid_diag)
                
                @test evolved_weighted_mass ≈ 1.0 atol = 1e-10
                
                println("2D Non-uniform grid (k,z) - Initial weighted mass: ", initial_weighted_mass)
                println("2D Non-uniform grid (k,z) - Evolved weighted mass: ", evolved_weighted_mass)
                println("2D Grid dimensions: ", (Nk, Nz), " Total size: ", size(A_t, 1))
            end
            
            @testset "Multiple Time Steps" begin
                A_t, grid_diag, sm, (Nk, Nz) = create_2d_test_setup(uniform_grid = false, Nk = 50, Nz = 20)
                g = create_initial_distribution(size(A_t, 1), normalized = false)
                
                # Normalize by weighted mass
                g = normalize_by_weighted_mass(g, grid_diag)
                
                # Test that weighted mass is preserved over multiple time steps
                weighted_masses = Float64[]
                push!(weighted_masses, weighted_mass(g, grid_diag))
                
                for i in 1:10
                    g = GrowthModels.iterate_g(g, A_t, grid_diag, time_step = 0.1)
                    push!(weighted_masses, weighted_mass(g, grid_diag))
                end
                
                # All weighted masses should be approximately 1.0
                for mass in weighted_masses
                    @test mass ≈ 1.0 atol = 1e-10
                end
                
                println("2D Multiple time steps - Weighted masses: ", weighted_masses)
            end
        end
        
        @testset "2D States (k, z) with Poisson Process" begin
            @testset "Uniform Grid" begin
                A_t, grid_diag, sm, (Nk, Nz) = create_2d_poisson_test_setup(uniform_grid = true, Nk = 50)
                g_init = create_initial_distribution(size(A_t, 1), normalized = false)
                
                # Normalize by weighted mass (correct approach)
                g_init_normalized = normalize_by_weighted_mass(g_init, grid_diag)
                
                # Check initial mass
                initial_weighted_mass = weighted_mass(g_init_normalized, grid_diag)
                @test initial_weighted_mass ≈ 1.0 atol = 1e-10
                
                # Test evolution preserves weighted mass
                g_evolved = GrowthModels.iterate_g(g_init_normalized, A_t, grid_diag, time_step = 0.1)
                evolved_weighted_mass = weighted_mass(g_evolved, grid_diag)
                
                @test evolved_weighted_mass ≈ 1.0 atol = 1e-10
                
                println("2D Poisson Uniform grid (k,z) - Initial weighted mass: ", initial_weighted_mass)
                println("2D Poisson Uniform grid (k,z) - Evolved weighted mass: ", evolved_weighted_mass)
                println("2D Poisson Grid dimensions: ", (Nk, Nz), " Total size: ", size(A_t, 1))
            end
            
            @testset "Non-uniform Grid" begin
                A_t, grid_diag, sm, (Nk, Nz) = create_2d_poisson_test_setup(uniform_grid = false, Nk = 50)
                g_init = create_initial_distribution(size(A_t, 1), normalized = false)
                
                # Normalize by weighted mass (correct approach)
                g_init_normalized = normalize_by_weighted_mass(g_init, grid_diag)
                
                # Check initial mass
                initial_weighted_mass = weighted_mass(g_init_normalized, grid_diag)
                @test initial_weighted_mass ≈ 1.0 atol = 1e-10
                
                # Test evolution preserves weighted mass
                g_evolved = GrowthModels.iterate_g(g_init_normalized, A_t, grid_diag, time_step = 0.1)
                evolved_weighted_mass = weighted_mass(g_evolved, grid_diag)
                
                @test evolved_weighted_mass ≈ 1.0 atol = 1e-10
                
                println("2D Poisson Non-uniform grid (k,z) - Initial weighted mass: ", initial_weighted_mass)
                println("2D Poisson Non-uniform grid (k,z) - Evolved weighted mass: ", evolved_weighted_mass)
                println("2D Poisson Grid dimensions: ", (Nk, Nz), " Total size: ", size(A_t, 1))
            end
        end
        
        @testset "StateEvolution with 2D SolvedModel" begin
            # Test that the high-level StateEvolution interface works correctly for 2D
            A_t, grid_diag, sm, (Nk, Nz) = create_2d_test_setup(uniform_grid = false, Nk = 50, Nz = 20)
            g = create_initial_distribution(size(A_t, 1), normalized = false)
            
            # Normalize properly
            g_normalized = normalize_by_weighted_mass(g, grid_diag)
            
            # Test StateEvolution constructor that takes SolvedModel
            state_evolution = StateEvolution(g_normalized, sm, 5)
            
            @test isa(state_evolution, StateEvolution)
            @test size(state_evolution.S, 2) == 5  # 0, 1, 2, 3, 4, 5
            
            # Check that mass is conserved throughout evolution
            for i in 1:size(state_evolution.S, 2)
                mass_at_time_i = weighted_mass(state_evolution.S[:, i], grid_diag)
                println("2D StateEvolution mass at time ", i, ": ", mass_at_time_i)
                @test mass_at_time_i ≈ 1.0 atol = 1e-10
            end
            
            println("2D StateEvolution mass conservation check passed")
        end
    end
end