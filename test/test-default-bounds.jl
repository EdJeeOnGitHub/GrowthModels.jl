using Test, GrowthModels
@testset "Capital defaults follow resolved support" begin
    m=StochasticSkibaAbilityModel()
    wide=StateGrid(m;k=(n=80,),z=(n=12,width=6))
    narrow=StateGrid(m;k=(n=80,),z=(n=12,width=2))
    @test last(wide.k)>last(narrow.k)
    explicit=StateGrid(m;k=(n=80,bounds=(.01,5.)),z=(n=12,width=6))
    @test last(explicit.k)==5.
    high=ResourceModel(m;population=Population([1.,2.],[.3,.7]))
    higher=StateGrid(high;k=(n=80,),z=(n=12,width=6))
    @test last(higher.k)>last(wide.k)
    @test higher.specification.k.bounds == (first(higher.k),last(higher.k))
    # Refinement on a fixed selected support keeps bounds fixed and converges.
    refined=StateGrid(m;k=(n=160,),z=(n=12,width=2))
    @test first(refined.k)==first(narrow.k) && last(refined.k)==last(narrow.k)
    for g in (narrow,refined)
        fit=solve(m,g)
        @test fit.converged
        @test minimum(fit.diagnostics.boundary.lower_drift)>=-1e-10
        @test maximum(fit.diagnostics.boundary.upper_drift)<=1e-10
    end
end
