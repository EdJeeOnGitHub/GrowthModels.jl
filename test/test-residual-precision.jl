using Test, GrowthModels, SparseArrays
@testset "Stretched-grid residual precision" begin
    m=RamseyCassKoopmansModel()
    g=StateGrid(m;k=(n=900,bounds=(.001,20.),spacing=:stretched,power=3.9))
    fit=solve(m,g)
    @test fit.converged
    A=sparse(fit.generator);v=vec(copy(fit.value));c=vec(copy(fit.policy.consumption))
    reference=setprecision(256) do
        BigFloat.(A)*BigFloat.(v) .+ BigFloat.(c).^(1-BigFloat(m.γ))./(1-BigFloat(m.γ)) .- BigFloat(m.ρ).*BigFloat.(v)
    end
    @test maximum(abs,reference)<1e-6
    @test fit.diagnostics.residual_norm ≈ Float64(maximum(abs,reference)) atol=1e-12 rtol=1e-6
    @test fit.settings.residual_tolerance==1e-6
    # Stable action agrees with independent high-precision arithmetic even when
    # ordinary level products are ill-conditioned.
    stable=GrowthModels.generator_action(A,v)
    exact=setprecision(256) do
        BigFloat.(A)*BigFloat.(v)
    end
    @test maximum(abs.(stable.-exact))<1e-11
end
