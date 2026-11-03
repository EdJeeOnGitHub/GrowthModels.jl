# Run with: julia --project=. test/benchmark-performance.jl
# Optional first argument: a checkout of the pre-optimization package, loaded
# with this environment's dependencies for a like-for-like HJB comparison.
using LinearAlgebra, SparseArrays, Printf, Serialization, ForwardDiff
if isempty(ARGS)
    using GrowthModels
else
    include(joinpath(abspath(ARGS[1]), "src", "GrowthModels.jl"))
    using .GrowthModels
end

function measure(f; samples = 3)
    f() # compile before measuring
    runs = []
    for _ in 1:samples
        GC.gc()
        push!(runs, @timed f())
    end
    order = sortperm([r.time for r in runs])
    run = runs[order[cld(samples, 2)]]
    return run.value, run.time, run.bytes / 2.0^20
end

function uncached_path(g0, A_t, weights, steps)
    g = copy(g0)
    saved = Matrix{eltype(g)}(undef, length(g), steps + 1)
    saved[:, 1] .= g
    for t in 1:steps
        g = iterate_g(g, A_t, weights; time_step = 0.25)
        saved[:, t+1] .= g
    end
    return saved
end

function benchmark_performance()
    results = Dict{String,Any}()
    kh = HyperParams(N = 200, xmin = 0.01, xmax = 5.0, coef = 2.0, power = 3.0)
    zh = HyperParams(N = 21, xmin = 0.8, xmax = 1.2)
    for (name, m, hp) in (
        ("deterministic", SkibaModel(), StateSpaceHyperParams((k = kh,))),
        ("OU_RCK", StochasticRamseyCassKoopmansModel(), StateSpaceHyperParams((k = kh, z = zh))),
        ("ability", StochasticSkibaAbilityModel(), StateSpaceHyperParams((k = kh, z = zh,
            η = HyperParams(N = 3, xmin = 0.8, xmax = 1.2)))))
        cold, tc, bc = measure(() -> solve_HJB(m, hp; verbose = false))
        @assert cold.value.convergence_status
        warm, tw, bw = measure(() -> solve_HJB(m, hp; verbose = false, init_value = cold.value))
        @assert warm.value.convergence_status
        @printf("HJB %-13s cold %.6f s %.3f MiB (%d iterations); warm %.6f s %.3f MiB (%d iterations)\n",
            name, tc, bc, cold.iter, tw, bw, warm.iter)
        results[name] = (value = cold.value.v, consumption = cold.variables.c, cold_time = tc,
            cold_MiB = bc, warm_time = tw, warm_MiB = bw)
        name == "deterministic" && continue
        dims = size(cold.value.v)
        weights = create_dx_stacked(cold.variables.k[:, 1, 1], prod(dims) ÷ dims[1])
        A_t = sparse(cold.value.A')
        g0 = fill(1 / sum(weights), length(weights))
        reference, tr, br = measure(() -> uncached_path(g0, A_t, weights, 100))
        if isdefined(GrowthModels, :EvolutionWorkspace)
            full, tf, bf = measure(() -> StateEvolution(g0, A_t, collect(0.0:0.25:25.0), dims, weights))
            final, te, be = measure(() -> StateEvolution(g0, A_t, collect(0.0:0.25:25.0), dims, weights; save = :final))
            @assert maximum(abs.(full.S - reference)) < 1e-12
            @assert final.final_density ≈ full.final_density
            @printf("DENSITY %-9s uncached %.6f s %.3f MiB; cached %.6f s %.3f MiB (%.1fx); final-only %.6f s %.3f MiB\n",
                name, tr, br, tf, bf, tr/tf, te, be)
        end
    end
    function derivative_problem(p)
        n = 300
        A = spdiagm(-1 => fill(-one(p[1]), n-1), 0 => fill(4 + sum(p), n),
            1 => fill(-one(p[1]), n-1))
        b = collect(1:n) .* ((1 + p[1]) / n)
        return A \ b
    end
    jacobian, tj, bj = measure(() -> ForwardDiff.jacobian(derivative_problem, fill(0.1, 6)))
    @printf("DIFFERENTIATION 300 states, 6 directions: %.6f s %.3f MiB\n", tj, bj)
    results["derivative"] = (jacobian = jacobian, time = tj, MiB = bj)
    if length(ARGS) >= 2
        serialize(ARGS[2], results)
    end
    return results
end

benchmark_performance()
