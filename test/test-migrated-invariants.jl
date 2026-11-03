using Test, GrowthModels, SparseArrays
@testset "Migrated scientific invariants" begin
    for p in (OrnsteinUhlenbeckProcess(θ=.1,σ=.1),PoissonProcess(z=[-.4,0.],Q=[-.9 .9;.1 -.1]))
        model=StochasticSkibaModel(p)
        grid=StateGrid(model;k=(n=80,bounds=(.01,5.),spacing=:stretched,power=2.),
            z=p isa PoissonProcess ? p.z : collect(range(.8,1.2;length=8)))
        fit=solve(model,grid);require_converged(fit)
        raw=ones(size(grid));pdf=raw./sum(raw.*cell_weights(grid))
        initial=distribution(grid;density=pdf)
        @test sum(mass(initial)) ≈ 1
        for dt in (.1,.25)
            path=trajectory(Evolution(fit;dt=dt),initial,[0.,.13,1.,2.])
            @test all(t -> abs(sum(density(path,t).*cell_weights(grid))-1)<1e-10,path.times)
        end
        replay=solve(model,grid)
        @test replay.value == fit.value
        @test replay.policy.consumption == fit.policy.consumption
    end
    A=sparse([0. 0 1.;0 0 2.;0 0 -3.])
    initial=[0.,0.,1.];perm=[3,1,2]
    p=GrowthModels.stationary_mass(A;initial_mass=initial)
    @test GrowthModels.stationary_mass(A[perm,perm];initial_mass=initial[perm]) ≈ p[perm]
    @test GrowthModels.stationary_mass(spzeros(1,1)) == [1.]
    @test GrowthModels.stationary_mass(spzeros(2,2);initial_mass=[.3,.7]) == [.3,.7]
end
