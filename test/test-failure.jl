using Test, GrowthModels
@testset "Finite failure snapshots" begin
    model=RamseyCassKoopmansModel(γ=.03)
    grid=StateGrid(model;k=(n=100,))
    fit=solve(model,grid;Delta=1000,maxiters=100)
    @test !fit.converged
    @test fit.diagnostics.stop_reason == :nonfinite
    @test fit.diagnostics.iterations > 1
    @test all(isfinite,fit.value)
    @test all(isfinite,fit.policy.consumption)
    residual=fit.policy.consumption.^(1-model.γ)./(1-model.γ) .+
        reshape(fit.generator*vec(fit.value),size(grid)) .- model.ρ.*fit.value
    @test maximum(abs.(residual)) ≈ fit.diagnostics.residual_norm
    @test fit.generator*grid.k ≈ fit.policy.drift atol=1e-9
    # Recompute the returned policy from that exact finite value without stepping.
    replay=solve(model,grid;initial=fit.value,maxiters=0,Delta=1000)
    @test replay.policy.consumption == fit.policy.consumption
    @test replay.generator == fit.generator
    @test_throws ArgumentError solve(RamseyCassKoopmansModel(γ=.01),grid;initial=zeros(size(grid)))
end
