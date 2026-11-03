using GrowthModels, Test, LinearAlgebra, SparseArrays, ForwardDiff

@testset "Performance and workspace contracts" begin
    @testset "Evolution factorization reuse and invalidation" begin
        A = sparse([-1.0 2 0; 1 -3 1; 0 1 -1])
        weights = [0.5, 1.0, 2.0]
        g = [1.0, 0.2, 0.1]
        ws = EvolutionWorkspace(A, weights; time_step = 0.25)
        factor = ws.solver.factors[1]
        mass_buffer, next_buffer = ws.mass, ws.next_mass
        @test iterate_g(g, ws) ≈ iterate_g(g, A, weights; time_step = 0.25)
        @test iterate_g(g, A, weights; time_step = 0.25, workspace = ws) ≈ iterate_g(g, ws)
        @test ws.factorizations == 1
        @test ws.solver.factors[1] === factor
        @test ws.mass === mass_buffer || ws.mass === next_buffer
        @test ws.next_mass === mass_buffer || ws.next_mass === next_buffer
        # The cache owns a snapshot; mutating the caller's generator cannot
        # silently corrupt the already factored system.
        A[1, 1] = -2
        A[2, 1] = 2
        @test ws.A_t[1, 1] == -1
        updated = iterate_g(g, A, weights; time_step = 0.25, workspace = ws)
        @test updated ≈ iterate_g(g, A, weights; time_step = 0.25)
        @test ws.factorizations == 2
        @test ws.solver.factors[1] === factor # same symbolic pattern
        update_evolution!(ws, A; time_step = 0.5)
        @test ws.factorizations == 3
        @test iterate_g(g, ws) ≈ iterate_g(g, A, weights; time_step = 0.5)
        weights2 = [1.0, 2.0, 4.0]
        update_evolution!(ws, A; dx_stacked = weights2)
        @test ws.factorizations == 3
        @test iterate_g(g, ws) ≈ iterate_g(g, A, weights2; time_step = 0.5)
        # A new nonzero requires fresh symbolic analysis.
        A[3, 1] = 0.5
        A[1, 1] -= 0.5
        update_evolution!(ws, A)
        @test ws.factorizations == 4
        @test ws.solver.factors[1] !== factor
        @test iterate_g(g, ws) ≈ iterate_g(g, A, weights2; time_step = 0.5)
        @test_throws DimensionMismatch update_evolution!(ws, spzeros(2, 2))
        @test_throws ArgumentError update_evolution!(ws; time_step = 0)
        @test_throws ArgumentError EvolutionWorkspace(A, [1.0, 0, 1])

        block = sparse([-1.0 1; 1 -1])
        block_A = kron(sparse(I, 2, 2), block)
        blocked = EvolutionWorkspace(block_A, ones(4); blocksize = 2)
        @test length(blocked.solver.factors) == 2
        @test iterate_g([1.0, 0, 0, 0], blocked) ≈ [2/3, 1/3, 0, 0]
        block_A[3, 1] = 0.1
        block_A[1, 1] -= 0.1
        @test_throws ArgumentError update_evolution!(blocked, block_A)
        # Even a pre-existing stored zero must not hide a newly coupled block.
        B = sparse(I, 4, 4) * 1.0
        B[3, 1] = 1.0
        B[3, 1] = 0.0
        solver = GrowthModels.SparseBlockSolver(B; blocksize = 2)
        B[3, 1] = 0.1
        @test_throws ArgumentError GrowthModels.refactor!(solver, B)
    end

    @testset "Selective storage, streaming, real times, and stationary opt-in" begin
        A = sparse([-1.0 1; 1 -1])
        weights = [0.5, 2.0]
        g = [2.0, 0.0]
        original = copy(g)
        times = [0.0, 0.25, 0.75, 1.5]
        ws = EvolutionWorkspace(A, weights; time_step = 0.25)
        all_states = StateEvolution(g, A, times, (2,), weights; dt = 0.25, workspace = ws)
        final = StateEvolution(g, A, times, (2,), weights; dt = 0.25, save = :final)
        moments = Tuple{Float64,Float64}[]
        stream = StateEvolution(g, A, times, (2,), weights; dt = 0.25, save = :none,
            callback = (t, p) -> push!(moments, (t, dot([1.0, 3.0], p))))
        @test ws.factorizations == 1
        @test all_states.times == times
        @test all_states.stationary_dist === nothing
        @test size(all_states.S) == (2, 4)
        @test size(final.S) == (2, 1)
        @test final.times == [1.5]
        @test size(stream.S) == (2, 0)
        @test isempty(stream.cell_masses)
        @test isempty(stream.times)
        @test all_states.final_density ≈ final.final_density ≈ stream.final_density
        @test final.S[:, 1] ≈ all_states.S[:, end]
        @test final.final_time == stream.final_time == 1.5
        @test first.(moments) == times
        @test last.(moments) ≈ vec([1.0, 3.0]' * all_states.cell_masses)
        @test g == original
        @test all_states[0.75] == all_states.S[:, 3]
        @test_throws ArgumentError all_states[0.5]
        reference = copy(g)
        for _ in 1:6
            reference = iterate_g(reference, A, weights; time_step = 0.25)
        end
        @test all_states.final_density ≈ reference
        stationary = StateEvolution(g, A, times, (2,), weights; stationary = true)
        @test stationary.stationary_dist ≈ StationaryDistribution(all_states)
        @test stationary.stationary_dist ≈ StationaryDistribution(A, weights)
        scaled = StateEvolution(3g, A, times, (2,), weights; normalize = false)
        @test weighted_mass(scaled.final_density, weights) ≈ 3
        @test_throws ArgumentError StateEvolution(g, A, [0.0, 0.0], (2,), weights)
        @test_throws ArgumentError StateEvolution(g, A, times, (2,), weights; dt = -1)
        @test_throws ArgumentError StateEvolution(g, A, times, (2,), weights; save = :invalid)
        @test_throws ArgumentError StateEvolution(g, A, 0, (2,), weights)
        sparse_output = StateEvolution(g, A, 3, (2,), weights; implicit_steps = 100)
        @test size(sparse_output.S) == (2, 3) # no 300-column substep history
        @test sparse_output.times == [0, 1, 2]
        # Reusing a workspace does not mutate a previously returned trajectory.
        before = copy(all_states.S)
        StateEvolution(reverse(g), A, times, (2,), weights; workspace = ws, dt = 0.25)
        @test all_states.S == before
    end

    @testset "Warm starts, trace ownership, and structured HJB solves" begin
        kh = HyperParams(N = 100, xmin = 0.01, xmax = 5.0, coef = 2.0, power = 3.0)
        hp = StateSpaceHyperParams((k = kh,))
        for m in (RamseyCassKoopmansModel(), SkibaModel())
            cold = solve_HJB(m, hp; verbose = false, crit = 1e-9)
            reference = solve_HJB(m, hp; verbose = false, crit = 1e-9, linear_solver = :sparse)
            @test cold.value.convergence_status && reference.value.convergence_status
            @test cold.value.v ≈ reference.value.v atol = 1e-7
            @test cold.variables.c ≈ reference.variables.c atol = 1e-7
            saved = deepcopy(cold.value)
            warm = solve_HJB(m, hp; init_value = cold.value, verbose = false, crit = 1e-9)
            @test warm.iter == 1
            @test warm.value.v ≈ cold.value.v atol = 1e-8
            @test length(warm.value.dist) == warm.iter + 1
            @test length(cold.value.dist) == cold.iter + 1
            @test cold.value.v == saved.v && cold.value.dist == saved.dist
            @test cold.value.dVf == saved.dVf && cold.value.A == saved.A
            guess = fill(-7.0, 100)
            unchanged = solve_HJB(m, hp; init_value = guess, maxit = 0, verbose = false)
            @test unchanged.value.v == guess
            @test unchanged.value.v !== guess
            @test unchanged.variables === nothing
            @test !unchanged.value.convergence_status
            zeros_start = solve_HJB(m, hp; init_value = zeros(100), maxit = 0, verbose = false)
            @test all(iszero, zeros_start.value.v)
            @test_throws DimensionMismatch solve_HJB(m, hp; init_value = zeros(2), verbose = false)
            @test_throws ArgumentError solve_HJB(m, hp; init_value = fill(NaN, 100), verbose = false)
        end
        base_model = RamseyCassKoopmansModel()
        base = solve_HJB(base_model, hp; verbose = false, crit = 1e-9)
        changed = RamseyCassKoopmansModel(A = base_model.A * 1.01)
        changed_cold = solve_HJB(changed, hp; verbose = false, crit = 1e-9)
        changed_warm = solve_HJB(changed, hp; init_value = base.value, verbose = false, crit = 1e-9)
        @test changed_warm.iter <= changed_cold.iter
        @test changed_warm.variables.c ≈ changed_cold.variables.c atol = 1e-7
        @test changed_warm.value.v ≈ changed_cold.value.v atol = 1e-7
        p = PoissonProcess(z = [0.0, 0.1], Q = [-0.2 0.2; 0.3 -0.3])
        m = StochasticSkibaAbilityModel(p)
        hp = StateSpaceHyperParams((k = kh, z = HyperParams(N = 2, xmin = 0.0, xmax = 0.1),
            η = HyperParams(N = 3, xmin = 0.8, xmax = 1.2)))
        blocked = solve_HJB(m, hp; verbose = false)
        whole = solve_HJB(m, hp; verbose = false, linear_solver = :sparse)
        @test blocked.value.convergence_status && whole.value.convergence_status
        @test blocked.value.v ≈ whole.value.v atol = 1e-7
        @test blocked.variables.c ≈ whole.variables.c atol = 1e-7
        @test Matrix(blocked.value.A) ≈ Matrix(whole.value.A) atol = 1e-6
        ws = GrowthModels.HJBWorkspace(m, StateSpace(m, hp), hp)
        GrowthModels.prepare_hjb_step!(ws, m, blocked.value.v, 1000)
        solver = GrowthModels.hjb_linear_solver(ws, m, :auto)
        @test length(solver.blocks) == 3
        @test all(size(b) == (200, 200) for b in solver.blocks)
        rates, rhs, consumption = nonzeros(ws.A), ws.rhs, ws.c
        pattern = (copy(ws.A.colptr), copy(rowvals(ws.A)))
        # A very different value changes policies, but not storage or pattern.
        GrowthModels.prepare_hjb_step!(ws, m, zeros(size(blocked.value.v)), 1000)
        @test nonzeros(ws.A) === rates && ws.rhs === rhs && ws.c === consumption
        @test (ws.A.colptr, rowvals(ws.A)) == pattern
        factors = copy(solver.factors)
        GrowthModels.refactor!(solver, ws.B)
        @test all(solver.factors[i] === factors[i] for i in eachindex(factors))
    end

    @testset "Sparse differentiation shares the primal system" begin
        function sparse_solution(p)
            A = sparse([3+p[1] -one(p[1]) zero(p[1]);
                -one(p[1]) 4+p[2] -one(p[1])/2;
                zero(p[1]) -one(p[1])/2 2+p[3]])
            b = [1+p[1]*p[2], 2-p[3], 1+p[2]]
            return A \ b
        end
        p = [0.2, 0.3, 0.4]
        derivative = ForwardDiff.jacobian(sparse_solution, p)
        h = 1e-6
        finite_difference = hcat([(sparse_solution(p + h .* [j == i for j in 1:3]) - sparse_solution(p - h .* [j == i for j in 1:3])) / (2h) for i in 1:3]...)
        @test derivative ≈ finite_difference rtol = 1e-7 atol = 1e-9
        gradient = p -> ForwardDiff.gradient(p -> sum(sparse_solution(p)), p)
        hessian = ForwardDiff.hessian(p -> sum(sparse_solution(p)), p)
        finite_hessian = hcat([(gradient(p + h .* [j == i for j in 1:3]) -
            gradient(p - h .* [j == i for j in 1:3])) / (2h) for i in 1:3]...)
        @test hessian ≈ finite_hessian rtol = 1e-6 atol = 1e-8
        # Refactorization must refresh both primal values and tangent data.
        function refactored_solution(p)
            A = sparse([p[1]+2 -one(p[1]); -one(p[1]) p[2]+3])
            factor = GrowthModels.factor_sparse(A)
            B = copy(A)
            B[1, 1] += p[2]
            GrowthModels.refactor_sparse!(factor, B)
            return GrowthModels.solve_sparse_factor(factor, [p[1], p[2]])
        end
        @test ForwardDiff.jacobian(refactored_solution, p[1:2]) ≈
            hcat([(refactored_solution(p[1:2] + h .* [j == i for j in 1:2]) -
                refactored_solution(p[1:2] - h .* [j == i for j in 1:2])) / (2h) for i in 1:2]...) rtol = 1e-7
        # A real RHS must contribute zero tangents without a special path.
        @test ForwardDiff.derivative(x -> (sparse([x 0; 0 x]) \ [1.0, 2.0])[1], 2.0) ≈ -0.25
    end
end

function differentiable_hjb_policy(p; stochastic = false, linear_solver = :auto, initial_value = nothing, return_result = false)
    T = eltype(p)
    kh = HyperParams(N = 40, xmin = T(0.05), xmax = T(5), coef = T(0), power = T(0))
    if stochastic
        process = PoissonProcess(z = T.([0.0, 0.1]), Q = T.([-0.2 0.2; 0.3 -0.3]))
        m = StochasticRamseyCassKoopmansModel(T(2), T(0.3), p[2], T(0.05), p[1], process)
        hp = StateSpaceHyperParams((k = kh, z = HyperParams(N = 2, xmin = T(0), xmax = T(0.1))))
    else
        m = RamseyCassKoopmansModel(T(2), T(0.3), p[2], T(0.05), p[1])
        hp = StateSpaceHyperParams((k = kh,))
    end
    result = solve_HJB(m, hp; verbose = false, crit = 1e-10, linear_solver = linear_solver, init_value = initial_value)
    @assert result.value.convergence_status
    return return_result ? result : vec(result.variables.c)[[10, 20, 30]]
end

@testset "HJB derivatives through structured and sparse solves" begin
    p = [0.6, 0.05]
    for stochastic in (false, true)
        f = p -> differentiable_hjb_policy(p; stochastic = stochastic)
        derivative = ForwardDiff.jacobian(f, p)
        h = 1e-6
        finite_difference = hcat([(f(p + h .* [i == 1, i == 2]) - f(p - h .* [i == 1, i == 2])) / (2h) for i in 1:2]...)
        @test derivative ≈ finite_difference rtol = 1e-5 atol = 1e-7
        reference = ForwardDiff.jacobian(p -> differentiable_hjb_policy(p;
            stochastic = stochastic, linear_solver = :sparse), p)
        @test derivative ≈ reference rtol = 1e-8 atol = 1e-9
        initial = differentiable_hjb_policy(p; stochastic = stochastic, return_result = true).value
        warm_derivative = ForwardDiff.jacobian(p -> differentiable_hjb_policy(p;
            stochastic = stochastic, initial_value = initial), p)
        @test warm_derivative ≈ derivative rtol = 1e-8 atol = 1e-9
    end
end

@testset "Differentiable cached grid geometry" begin
    grid_sum = power -> sum(GrowthModels.generate_grid(8, 0.1, 5.0, 2.0, power))
    h = 1e-6
    @test ForwardDiff.derivative(grid_sum, 3.0) ≈ (grid_sum(3+h)-grid_sum(3-h))/(2h) rtol = 1e-6
    grid = GrowthModels.generate_grid(8, 0.1, 5.0, 2.0, 0.0)
    @test first(grid) ≈ 0.1
    @test last(grid) ≈ 5.0
    @test GrowthModels.generate_dx(view(grid, :))[1] ≈ fill((5-0.1)/7, 8)
end
