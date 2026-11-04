using Test, GrowthModels, GrowthCounterfactuals
using GrowthCounterfactuals.ModelSolver: ForwardDiff
const MS = GrowthCounterfactuals.ModelSolver
const PI = GrowthCounterfactuals.ProdFunctionInterp
const TB = GrowthCounterfactuals.TransferBasisFunctions

@testset "GrowthCounterfactuals standard API integration" begin
    @test realpath(dirname(pathof(GrowthModels))) == realpath(joinpath(@__DIR__, "..", "..", "src"))
    @test !isdefined(GrowthModels,:update_v)
    @test !isdefined(GrowthModels,:solve_HJB)
    k=collect(range(.01,10.;length=200))
    interpolant=PI.interp_prod_function_factory(k,.6 .* k.^.3,1.;quantile_cutoffs=[0.,1.])
    empirical=hcat(collect(range(.1,5.;length=100)),ones(100),vcat(fill(.8,30),fill(1.2,70)))
    ranges=Dict(:k=>(.01,8.),:ω=>(.8,1.2),:η=>(.8,1.2))
    proc=Dict(:OU_theta=>.1,:OU_sigma=>.1)
    sp=MS.SolverParams(grid_dim=(80,8,2),crit=1e-7,pivot=2.)
    mp=MS.ModelParams(interpolant,(k,z,a)->.2k,ranges,empirical,proc,zeros(100,1),ones(100),sp)
    @test mp.population.shares == [.3,.7]
    transfers=MS.TransferPolicies(flat_transfer=.1,min_y=.3)
    fit=MS.generate_solved_model([2.,.05,.05],mp,sp;transfers=transfers)
    @test fit.converged
    @test which(solve,(typeof(fit.model),typeof(fit.grid))).module === GrowthModels
    @test resources(fit.model,(k=.001,z=mp.grid.z[1],ability=.8)) == .3
    @test minimum(fit.policy.resources) >= .3
    @test fit.diagnostics.residual_norm < sp.crit
    same=MS.generate_solved_model([2.,.05,.05],mp,sp;initial=fit,transfers=transfers)
    @test same.diagnostics.iterations == 1
    changed=MS.generate_solved_model([2.,.051,.05],mp,sp;initial=fit,transfers=transfers)
    cold_changed=MS.generate_solved_model([2.,.051,.05],mp,sp;transfers=transfers)
    @test changed.converged && cold_changed.converged
    @test changed.value ≈ cold_changed.value atol=1e-5
    @test changed.policy.consumption ≈ cold_changed.policy.consumption atol=1e-5
    # Volatility moves the productivity coordinates; explicitly rebuild and
    # regrid the value guess rather than reusing stale coordinates or policies.
    narrower=ResourceModel(FunctionalModel(interpolant,
        OrnsteinUhlenbeckProcess(θ=.1,σ=.095),mp.population;γ=2.,ρ=.05,δ=.05);
        transfer=.1,income_floor=.3)
    next_grid=StateGrid(narrower;k=mp.grid.k,
        z=(n=8,bounds=(mp.grid.z[1]^.95,mp.grid.z[end]^.95)))
    moved=GrowthModels.solve(narrower,next_grid;initial=regrid(fit,next_grid),
        value_tolerance=sp.crit,residual_tolerance=sp.crit)
    cold_moved=GrowthModels.solve(narrower,next_grid;
        value_tolerance=sp.crit,residual_tolerance=sp.crit)
    @test moved.converged && cold_moved.converged
    @test moved.value ≈ cold_moved.value atol=1e-5
    @test moved.policy.consumption ≈ cold_moved.policy.consumption atol=1e-5
    @test TB.create_value_function_interpolation(fit)(mp.grid.k[20],mp.grid.z[2],.8) ≈ fit.value[20,2,1]
    d=distribution(mp.grid;masses=mp.grid_weights)
    @test sum(mass(stationary(fit,d))) ≈ 1
    function stationary_moment(rho)
        pars=[one(rho)*2,rho,one(rho)*.05]
        r=MS.generate_solved_model(pars,mp,sp;initial=fit,transfers=transfers)
        require_converged(r)
        return expectation(stationary(r,d),state->state.k)
    end
    h=1e-5
    ad=ForwardDiff.derivative(stationary_moment,.05)
    fd=(stationary_moment(.05+h)-stationary_moment(.05-h))/(2h)
    @test isfinite(ad)
    @test ad ≈ fd rtol=1e-3 atol=1e-5
end
