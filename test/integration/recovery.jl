using Test, GrowthModels, JointModel, Random
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))
include(joinpath(analysis_root, "code/growth-models/AnalysisPolicy.jl"))
using .AnalysisPolicy
const example = joinpath(analysis_root, "code/joint-generative-model/examples/simulate-fit-recovery.jl")
# Run the example's real DGP functions without its unrelated estimator/plot job.
for expr in Meta.parseall(read(example, String); filename=example).args
    if expr isa Expr && expr.head == :function
        signature = expr.args[1]
        if signature isa Expr && signature.head == :call &&
                signature.args[1] in (:solve_rck_ability, :simulate_structural_panels)
            Core.eval(@__MODULE__, expr)
        end
    end
end

@testset "Continuous-shock recovery example migration" begin
    structural = solve_rck_ability()
    fit = structural.solved
    @test fit.converged
    @test fit.grid.k ≈ collect(range(.05, 8.; length=100))
    @test structural.policy(1., 1., fit.grid.ability[4]) ≈
        consumption(fit, (k=1., z=1., ability=fit.grid.ability[4]))
    bounded = continuous_ability_policy(fit)
    @test_throws BoundsError bounded(1., 1., 1.5)
    # Independent endpoint-slope check for the explicitly retained boundary rule.
    lo, hi = fit.grid.ability[end-1:end]
    cl = consumption(fit, (k=1., z=1., ability=lo))
    ch = consumption(fit, (k=1., z=1., ability=hi))
    @test structural.policy(1., 1., 1.5) ≈ ch + (1.5-hi)*(ch-cl)/(hi-lo)
    simulated = simulate_structural_panels(Xoshiro(20260918), structural;
        households=120, visits=5)
    @test length(simulated.panels) == 120
    @test count(simulated.treated) == 60
    @test all(p -> all(isfinite, p.k) && all(isfinite, p.y) && all(isfinite, p.c), simulated.panels)
end
