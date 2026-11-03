using GrowthModels
using Test
using SparseArrays
using LinearAlgebra

@testset "P1 numerical correctness" begin
    @testset "Density evolution preserves mass without renormalizing" begin
        A = sparse([-1.0 1.0; 1.0 -1.0])
        weights = [1.0, 2.0]
        g = [1.0, 0.0]
        @test iterate_g(g, A', weights) ≈ [2/3, 1/6]
        @test iterate_g(3g, A', weights) ≈ 3 .* [2/3, 1/6]
        @test weighted_mass(iterate_g(3g, A', weights), weights) ≈ 3
        copy_g = copy(g)
        iterate_g!(copy_g, A', weights)
        @test copy_g ≈ [2/3, 1/6]
        @test g == [1.0, 0.0]
        # The unweighted overload operates on masses, not densities.
        @test iterate_g(weights .* g, A') ≈ weights .* copy_g

        stationary = StationaryDistribution(A, weights)
        @test stationary ≈ [1/2, 1/4]
        @test iterate_g(stationary, A, weights) ≈ stationary
        @test norm(A * (weights .* stationary), Inf) < 1e-12
    end

    @testset "Stationary classes and absorption of transient mass" begin
        block = sparse([-1.0 1.0; 1.0 -1.0])
        A_t = kron(sparse(I, 2, 2), block)
        weights = [0.1, 2.0, 0.3, 4.0]
        p0 = [0.2, 0.1, 0.6, 0.1]
        g0 = p0 ./ weights
        @test_throws ArgumentError StationaryDistribution(A_t, weights)
        stationary = StationaryDistribution(A_t, weights; initial_density = g0)
        @test weights .* stationary ≈ [0.15, 0.15, 0.35, 0.35]
        @test iterate_g(stationary, A_t, weights) ≈ stationary
        for _ in 1:20
            g0 = iterate_g(g0, A_t, weights; time_step = 0.2)
            @test sum((weights .* g0)[1:2]) ≈ 0.3 atol = 1e-12
            @test sum((weights .* g0)[3:4]) ≈ 0.7 atol = 1e-12
            @test minimum(g0) >= 0
        end
        evolution = StateEvolution(p0 ./ weights, A_t, 3, (2, 2), weights; stationary = true)
        @test evolution.stationary_dist ≈ stationary
        for t in axes(evolution.S, 2)
            @test vec(sum(evolution.cell_masses[:, :, t], dims = 1)) ≈ [0.3, 0.7]
        end

        # State 1 exits to absorbing states 2 and 3 at rates 1 and 3.
        # The two absorbing classes are weakly connected through the transient state.
        A_t = sparse([-4.0 0 0; 1 0 0; 3 0 0])
        weights = [0.5, 2.0, 3.0]
        g0 = [0.4, 0.1, 0.5] ./ weights
        stationary = StationaryDistribution(A_t, weights; initial_density = g0)
        @test weights .* stationary ≈ [0, 0.2, 0.8]
        @test norm(A_t * (weights .* stationary), Inf) < 1e-12
        # State order must not affect the decomposition or absorption weights.
        perm = [3, 1, 2]
        @test StationaryDistribution(A_t[perm, perm], weights[perm];
            initial_density = g0[perm]) ≈ stationary[perm]
        @test StationaryDistribution(spzeros(1, 1), [2.0]) ≈ [0.5]
        @test StationaryDistribution(spzeros(2, 2), [1, 2];
            initial_density = [1, 1]) ≈ [1/3, 1/3]
        # A unique closed class does not require a specified initial density.
        @test StationaryDistribution(sparse([-2.0 0; 2 0]), [1.0, 2.0]) ≈ [0, 0.5]
        # Two transient states can exchange mass before absorption.
        transient_generator = sparse([-3.0 1 2 0; 2 -5 0 3; 0 0 0 0; 0 0 0 0])
        absorbed = StationaryDistribution(sparse(transient_generator'), ones(4);
            initial_density = [1.0, 0, 0, 0])
        @test absorbed ≈ [0, 0, 10/13, 3/13]
        @test StationaryDistribution(spzeros(3, 3), weights;
            initial_density = g0) ≈ g0
        @test_throws ArgumentError StationaryDistribution(A_t, weights; initial_density = zeros(3))
        @test_throws ArgumentError StationaryDistribution(A_t, weights; initial_density = [-1., 1., 1.])
        @test_throws DimensionMismatch StationaryDistribution(A_t, [1.0])
    end

    @testset "Exclusive Hamiltonian choice at nonconcave points" begin
        # Interior points cover forward, backward, tie, and zero-drift choices.
        dVf = [0.25, 4.0, 2.25, 2.25, 0.25, 4.0]
        dVb = [0.25, 0.25, 0.01, 0.25, 4.0, 4.0]
        policy = GrowthModels.select_upwind_policy(dVf, dVb, ones(6), 2.0)
        @test policy.If == [false, true, false, true, false, false]
        @test policy.Ib == [false, false, true, false, false, false]
        @test all(policy.If .+ policy.Ib .+ policy.I0 .== 1)
        @test policy.c ≈ [1, 0.5, 10, 2/3, 1, 1]
        @test policy.If .* policy.sf .+ policy.Ib .* policy.sb ≈ 1 .- policy.c
    end

    @testset "Stochastic update uses the selected drift in its generator" begin
        process = PoissonProcess(z = [1.0, 2.0], Q = [-0.2 0.2; 0.3 -0.3])
        k = [1.0, 2.0, 3.0]
        z = process.z
        for ability in (false, true)
            m = ability ? StochasticSkibaAbilityModel(process) : StochasticSkibaModel(process)
            kh = HyperParams(N = 3, xmin = 1.0, xmax = 3.0)
            zh = HyperParams(N = 2, xmin = 1.0, xmax = 2.0)
            hp = ability ? StateSpaceHyperParams((k = kh, z = zh, η = zh)) :
                StateSpaceHyperParams((k = kh, z = zh))
            dims = ability ? (3, 2, 2) : (3, 2)
            kk = repeat(reshape(k, 3, 1), 1, dims[2:end]...)
            y = 1 .+ m.δ .* kk
            state = ability ? StateSpace((k = k, z = z, η = z), (y = y,)) :
                StateSpace((k = k, z = z), (y = y,))
            value = Value(state)
            value.v[:] = vec(repeat(reshape([0.0, 0.25, 4.25], 3, 1), 1, dims[2:end]...))
            diffusion = GrowthModels.construct_diffusion_matrix(process, state, hp)
            output, _, variables = update_v(m, value, state, hp, diffusion; crit = Inf, verbose = false)
            @test all(variables.If .+ variables.Ib .+ variables.I0 .== 1)
            @test all(selectdim(variables.If, 1, 2))
            @test output.A * vec(kk) ≈ vec(y .- m.δ .* kk .- variables.c) atol = 1e-12
            @test GrowthModels.check_generator(output.A) === nothing
        end
    end

    @testset "OU generator is monotone and approximates the Ito drift" begin
        for σ in (0.0, 0.01, 0.1), Nz in (5, 21)
            process = OrnsteinUhlenbeckProcess(θ = 0.1, σ = σ)
            k = [1.0, 2.0]
            z = collect(range(0.8, 1.2, length = Nz))
            hp = StateSpaceHyperParams((k = HyperParams(N = 2, xmin = 1.0, xmax = 2.0),
                z = HyperParams(N = Nz, xmin = 0.8, xmax = 1.2)))
            state = StateSpace((k = k, z = z), (y = ones(2, Nz),))
            A = GrowthModels.construct_diffusion_matrix(process, state, hp)
            @test GrowthModels.check_generator(A) === nothing
            @test maximum(abs, sum(A, dims = 2)) < 1e-12
            rows, cols, vals = findnz(A)
            @test all(vals[rows .!= cols] .>= 0)
            zz = repeat(z, inner = 2)
            μ = (-process.θ .* log.(zz) .+ σ^2 / 2) .* zz
            @test (A * zz)[3:end-2] ≈ μ[3:end-2] atol = 1e-12
            # Upwinding adds |μ| Δz numerical variance, which vanishes as
            # the grid is refined; the physical variance is σ²z².
            expected_second_moment = 2 .* zz .* μ .+ σ^2 .* zz.^2 .+ abs.(μ) .* (z[2] - z[1])
            @test (A * zz.^2)[3:end-2] ≈ expected_second_moment[3:end-2] atol = 1e-12
            p = iterate_g(fill(1 / length(zz), length(zz)), A'; time_step = 10)
            @test minimum(p) >= 0
            @test sum(p) ≈ 1 atol = 1e-12
        end
        @test_throws ValueFunctionError GrowthModels.check_generator(sparse([1.0 -1.0; 1.0 -1.0]))
    end

    @testset "Zero-shock Skiba limit and fixed-ability distributions" begin
        k_hps = HyperParams(N = 120, xmin = 0.01, xmax = 5.0, coef = 2.0, power = 3.0)
        z_hps = HyperParams(N = 2, xmin = 0.0, xmax = 0.1)
        hp = StateSpaceHyperParams((k = k_hps, z = z_hps))
        process = PoissonProcess(z = [0.0, 0.1], Q = zeros(2, 2))
        deterministic = solve_HJB(SkibaModel(), StateSpaceHyperParams((k = k_hps,)); verbose = false)
        stochastic = solve_HJB(StochasticSkibaModel(process), hp; verbose = false)
        @test deterministic.value.convergence_status
        @test stochastic.value.convergence_status
        @test stochastic.value.v[:, 1] ≈ deterministic.value.v atol = 1e-7
        @test stochastic.variables.c[:, 1] ≈ deterministic.variables.c atol = 1e-7

        process = PoissonProcess(z = [0.0, 0.1], Q = [-0.2 0.2; 0.3 -0.3])
        η_hps = HyperParams(N = 2, xmin = 0.8, xmax = 1.2)
        hp = StateSpaceHyperParams((k = k_hps, z = z_hps, η = η_hps))
        m = StochasticSkibaAbilityModel(process)
        result = solve_HJB(m, hp; verbose = false)
        @test result.value.convergence_status
        sm = SolvedModel(m, result)
        weights = create_dx_stacked(sm.variables.k[:, 1, 1], 4)
        p0 = vcat(fill(0.3 / 240, 240), fill(0.7 / 240, 240))
        evolution = StateEvolution(p0 ./ weights, sm, 5; stationary = true)
        for t in 1:5
            @test vec(sum(evolution.cell_masses[:, :, :, t], dims = (1, 2))) ≈ [0.3, 0.7] atol = 1e-10
        end
        stationary_mass = reshape(evolution.stationary_dist .* weights, 120, 2, 2)
        @test vec(sum(stationary_mass, dims = (1, 2))) ≈ [0.3, 0.7] atol = 1e-10
        @test norm(sm.value.A' * vec(stationary_mass), Inf) < 1e-10
        @test minimum(stationary_mass) >= -1e-12
    end
end
