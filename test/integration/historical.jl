using Test, GrowthModels, SHA
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))
const adapter = joinpath(analysis_root, "sims/mainline/HistoricalMomentModel.jl")
module CurrentSource
    include(Main.adapter)
end
module ArchivedSource
    include(Main.adapter)
end
const current = CurrentSource.HistoricalMomentModel
const archived = ArchivedSource.HistoricalMomentModel
const source_path = joinpath(analysis_root, "code/joint-functions/test-joint-recovery.jl")
const archive_revision = "0d43c03d2841c1ec7395daade2acd40d58db8f6c"

current.load_historical(source_path)
mktemp() do path, io
    # Fixed historical Git source, not an edited archive or a compatibility shim.
    historical = read(`git -C $analysis_root show $archive_revision:code/joint-functions/test-joint-recovery.jl`, String)
    write(io, historical)
    close(io)
    hash_before = bytes2hex(sha256(read(path)))
    hash_loaded = archived.load_historical(path)
    Base.invokelatest() do
        @testset "Historical source adapter migration" begin
            @test hash_loaded == hash_before == bytes2hex(sha256(read(path)))
            @test occursin("GrowthModels.solve_HJB", historical)
            for make in (:rck_dgp, :skiba_dgp)
                dgp = current.override_joint_dgp(getproperty(current, make)(); k_grid_n=60)
                old_dgp = archived.override_joint_dgp(getproperty(archived, make)(); k_grid_n=60)
                fit = current.solve_growth_model(dgp)
                old_fit = archived.solve_growth_model(old_dgp)
                @test fit.converged && old_fit.converged
                @test fit.value ≈ old_fit.value atol=1e-11
                @test fit.grid.k ≈ collect(range(dgp.k_grid_min, dgp.k_grid_max; length=60))
                state = (k=2., z=1., ability=1.)
                @test production(fit.model, state) ≈ current.structural_production(dgp, 2., 0., 0.)
                truth, transfers, treated = current.simulate(fit, dgp, 8, 3, 42, 1.)
                old_truth, old_transfers, old_treated = archived.simulate(old_fit, old_dgp, 8, 3, 42, 1.)
                @test truth.k ≈ old_truth.k atol=1e-11
                @test truth.c ≈ old_truth.c atol=1e-11
                @test transfers == old_transfers && treated == old_treated
                @test all(isfinite, truth.k) && all(>(0), truth.c)
            end
        end
    end
end
