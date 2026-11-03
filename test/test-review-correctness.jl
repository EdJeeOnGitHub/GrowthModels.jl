using Test, GrowthModels, LinearAlgebra, SparseArrays, ForwardDiff
using GrowthModels: production_function_prime

@testset "Review correctness regressions" begin
    @testset "Residual convergence and low risk aversion" begin
        for gamma in (0.3, 0.5, 0.9, 1.5, 2.0)
            m = RamseyCassKoopmansModel(γ = gamma)
            hp = StateSpaceHyperParams(m; Nk = 150)
            r = solve_HJB(m, hp; crit = 1e-8, verbose = false)
            @test r.value.convergence_status
            @test maximum(abs.(GrowthModels.V_err(m)(r.value, r.variables))) < 1e-8
            @test maximum(r.variables.c) < 10
        end
        for gamma in (0.3, 0.5, 0.9)
            m = RamseyCassKoopmansModel(γ = gamma)
            r = solve_HJB(m, StateSpaceHyperParams(m; Nk = 500); Delta = 1000, maxit = 100, verbose = false)
            @test !r.value.convergence_status || (maximum(r.variables.c) < 10 &&
                maximum(abs.(GrowthModels.V_err(m)(r.value, r.variables))) < 1e-6)
        end
        m = RamseyCassKoopmansModel()
        @test_throws ArgumentError solve_HJB(RamseyCassKoopmansModel(γ = 1.0), StateSpaceHyperParams(m; Nk = 30); verbose = false)
        errors = Float64[]
        for n in (250, 1000)
            m = RamseyCassKoopmansModel(γ = 0.3, α = 0.3)
            kss = k_steady_state(m)
            hp = StateSpaceHyperParams((k = HyperParams(N = n, xmin = 0.3kss, xmax = 1.5kss),))
            r = solve_HJB(m, hp; crit = 1e-9, maxit = 2000, verbose = false)
            @test r.value.convergence_status
            phi = (m.ρ + m.δ * (1 - m.α)) / m.α
            push!(errors, maximum(abs.(r.variables.c[4:end-3] .- phi .* r.variables.k[4:end-3])))
        end
        @test errors[2] < errors[1] < 1e-3
    end

    @testset "Default grids represent the process and high types" begin
        m = StochasticRamseyCassKoopmansModel()
        hp = StateSpaceHyperParams(m; Nk = 2, Nz = 400)
        p = m.stochasticprocess
        sigma = p.σ / sqrt(2p.θ)
        @test log(hp[:z].xmin) ≈ -4sigma
        @test log(hp[:z].xmax) ≈ 4sigma
        st = StateSpace(m, hp)
        Q = GrowthModels.construct_diffusion_matrix(p, st, hp)[1:2:end, 1:2:end]
        mass = StationaryDistribution(sparse(Q'), ones(400))
        z = st[:z]
        @test dot(z, mass) ≈ exp(sigma^2/2) atol = 0.002
        @test sqrt(dot(log.(z).^2, mass) - dot(log.(z), mass)^2) ≈ sigma atol = 0.005
        a = StochasticSkibaAbilityModel()
        ahp = StateSpaceHyperParams(a; Nk = 50, Nz = 10, Nη = 3)
        @test production_function_prime(a, ahp[:k].xmax, ahp[:z].xmax, ahp[:η].xmax) < a.ρ + a.δ
        pp = PoissonProcess(z = [-0.01, 0.1, 0.2], Q = [-2.0 1 1; 1 -2 1; 1 1 -2])
        for pm in (StochasticSkibaModel(pp), StochasticRamseyCassKoopmansModel(pp), StochasticSkibaAbilityModel(pp))
            php = StateSpaceHyperParams(pm; Nk = 30)
            @test php[:z].N == 3
            @test StateSpace(pm, php)[:z] == pp.z
            @test size(php)[1:2] == (30, 3)
        end
        @test_throws DimensionMismatch StateSpaceHyperParams(StochasticSkibaModel(pp); Nz = 2)
        # Additive income has zero capital derivative: it cannot be added to k*.
        m1 = StochasticSkibaModel(pp)
        m2 = StochasticSkibaModel(PoissonProcess(z = pp.z .+ 1, Q = pp.Q))
        @test k_steady_state(m1) == k_steady_state(m2)
    end

    @testset "Evolution defaults, mass and switching trailing states" begin
        A = sparse([-0.3 0.2; 0.3 -0.2])
        g = [2.0, 0.0]
        coarse = StateEvolution(g, A, [0.0, 20.0], (2,), ones(2); normalize = false, stationary = true)
        fine = StateEvolution(g, A, collect(0.0:0.1:20.0), (2,), ones(2); normalize = false)
        @test coarse.final_density ≈ fine.final_density atol = 1e-12
        @test sum(coarse.stationary_dist) ≈ 2
        @test StationaryDistribution(coarse) ≈ coarse.stationary_dist
        ws = EvolutionWorkspace(A, ones(2); time_step = 0.25)
        StateEvolution(g, A, [0.0, 1.0], (2,), ones(2); dt = 0.3, workspace = ws)
        @test ws.factorizations == 1 # actual first substep is 0.25
        # A trailing coordinate is allowed to switch; :auto must fall back.
        Q = sparse([-1.0 1; 1 -1])
        coupled = kron(Q, sparse(I, 4, 4))
        path = StateEvolution(ones(8), coupled, [0.0, 0.5], (2, 2, 2), ones(8))
        @test sum(path.final_density) ≈ 1
        hp = StateSpaceHyperParams((k = HyperParams(N = 2, xmin = 0.1, xmax = 1.0),
            z = HyperParams(N = 2, xmin = 0.8, xmax = 1.2), η = HyperParams(N = 2, xmin = 0.8, xmax = 1.2)))
        m = StochasticSkibaAbilityModel()
        st = StateSpace(m, hp)
        ws_hjb = GrowthModels.HJBWorkspace(m, st, hp; diffusion_matrix = coupled)
        GrowthModels.prepare_hjb_step!(ws_hjb, m, GrowthModels.initial_guess(m, st), 1000)
        solver = GrowthModels.hjb_linear_solver(ws_hjb, m, :auto)
        out = zeros(8)
        GrowthModels.solve_linear!(out, solver, ws_hjb.rhs)
        @test out ≈ ws_hjb.B \ ws_hjb.rhs
    end

    @testset "Production derivatives and named crossings" begin
        m = SkibaModel()
        @test technology_crossing(m) == k_star(m)
        for k in (0.1, 2.2, 3.0, 5.0)
            @test production_function_prime(m, k) ≈ ForwardDiff.derivative(x -> production_function(m, x), k)
        end
        for p in (OrnsteinUhlenbeckProcess(θ = 0.1, σ = 0.1), PoissonProcess(z = [-0.5, 0.1], Q = [-1.0 1; 1 -1]))
            sm, am = StochasticSkibaModel(p), StochasticSkibaAbilityModel(p)
            for k in (0.1, 2.2, 3.0), z in (-0.5, 0.8)
                p isa OrnsteinUhlenbeckProcess && z < 0 && continue
                @test production_function_prime(sm, k, z) ≈ ForwardDiff.derivative(x -> production_function(sm, x, z), k)
                @test production_function_prime(am, k, z, 1.2) ≈ ForwardDiff.derivative(x -> production_function(am, x, z, 1.2), k)
            end
            @test all(isfinite, production_function_prime(sm, [0.1, 2.2, 3.0], 0.8))
        end
        m = SmoothSkibaModel(β = 500.0)
        for k in (0.01, 2.0, 3.0)
            @test isfinite(production_function_prime(m, k))
            @test production_function_prime(m, k) ≈ ForwardDiff.derivative(x -> production_function(m, x), k)
        end
        @test k_steady_state_hi(m) ≈ (m.α*m.A_H/(m.ρ+m.δ))^(1/(1-m.α))
    end

    @testset "Boundaries, transfers, wrappers, and fixed-grid AD" begin
        hp = StateSpaceHyperParams((k = HyperParams(N = 60, xmin = 0.1, xmax = 5.0),))
        m = RamseyCassKoopmansModel()
        r = solve_HJB(m, hp; flat_transfer = 0.2, verbose = false)
        sm = SolvedModel(m, r)
        k = r.variables.k[20]
        @test sm.kdot_function(k) ≈ r.variables.y[20] - m.δ*k - r.variables.c[20]
        @test sm.ydot_function(k) ≈ production_function_prime(m, k) * sm.kdot_function(k)
        # Stochastic diagnostics include the productivity generator, and
        # interpolated capital drift includes the transfer in every model.
        pp = PoissonProcess(z = [0.0, 0.1], Q = [-1.0 1; 1 -1])
        for stochastic in (StochasticSkibaModel(pp), StochasticSkibaAbilityModel(pp))
            shp = StateSpaceHyperParams(stochastic; Nk = 50, Nη = 2)
            fit = solve_HJB(stochastic, shp; flat_transfer = 0.2, verbose = false)
            @test fit.value.convergence_status
            @test maximum(abs.(GrowthModels.V_err(stochastic)(fit.value, fit.variables))) < 1e-6
            solved = SolvedModel(stochastic, fit)
            coords = stochastic isa StochasticSkibaAbilityModel ?
                (fit.variables.k[20], fit.variables.z[20], fit.variables.η[20]) :
                (fit.variables.k[20], fit.variables.z[20])
            @test solved.kdot_function(coords...) ≈ fit.variables.y[20] - stochastic.δ * coords[1] - fit.variables.c[20]
        end
        bad = solve_HJB(m, hp; maxit = 1, verbose = false)
        @test_throws ArgumentError SolvedModel(m, bad)
        @test_throws ArgumentError SolvedModel(m, bad.value, bad.variables)
        _, wrapped = solve_growth_model(m, (; Nk = 60))
        direct = solve_HJB(m, StateSpaceHyperParams(m; Nk = 60); verbose = false)
        @test wrapped.iter == direct.iter
        # A bad lower boundary in a middle shock state must be rejected.
        st = StateSpace((k = [0.1, 1.0], z = [0.0, 1.0, 2.0]), (y = [1.0 -0.1 1.0; 1.0 1.0 1.0],))
        @test st[CartesianIndex(1, 2)] == (k = 0.1, z = 1.0)
        @test_throws DomainError check_statespace_constraints(st, m)
        # Negative upper net output is admissible: capital can fall inward.
        highhp = StateSpaceHyperParams((k = HyperParams(N = 80, xmin = 0.1, xmax = 100.0),))
        high = solve_HJB(RamseyCassKoopmansModel(γ = 1.5), highhp; verbose = false)
        @test high.value.convergence_status
        @test all(isfinite, high.value.dVf)
        f = a -> sum(solve_HJB(RamseyCassKoopmansModel(promote(2.0, 0.3, 0.05, 0.05, a)...), hp; verbose = false, crit = 1e-8).variables.c)
        h = 1e-5
        @test ForwardDiff.derivative(f, 0.6) ≈ (f(0.6+h)-f(0.6-h))/(2h) rtol = 1e-5
        p = OrnsteinUhlenbeckProcess(θ = 0.1, σ = 0.1)
        for initial in (0.0, [0.0, 0.1])
            t, v = sample(p, initial, 1.0, 0.3; seed = 1)
            @test length(t) == (initial isa Number ? length(v) : size(v, 2))
        end
        t, v = sample(p, [0.0, 0.1], 1.0, 0.3, 3; seed = 1)
        @test length(t) == size(v, 2)
    end
end
