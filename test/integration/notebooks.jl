using Test, GrowthModels, DataFrames, StatsBase, Random, Interpolations
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))

"Load actual notebook functions without executing empirical/interactive cells."
function notebook_functions(relative, names)
    path = joinpath(analysis_root, "man/notebook", relative)
    sandbox = Module(gensym(:Notebook))
    Core.eval(sandbox, :(using GrowthModels, DataFrames, Statistics, StatsBase, Random))
    if :create_function_interpolation in names
        Base.include(sandbox, joinpath(analysis_root,"code/growth-models/AnalysisPolicy.jl"))
        Core.eval(sandbox, :(using .AnalysisPolicy, Interpolations))
    end
    for expr in Meta.parseall(read(path, String); filename=path).args
        if expr isa Expr && expr.head == :function
            signature = expr.args[1]
            if signature isa Expr && signature.head == :call && signature.args[1] in names
                Core.eval(sandbox, expr)
            end
        end
    end
    return sandbox
end

notebook_call(m, name, args...; kwargs...) =
    Base.invokelatest(() -> getproperty(m, name)(args...; kwargs...))

@testset "KKT notebook value and derivative profiles" begin
    model = ResourceModel(StochasticSkibaAbilityModel(); population=Population([.8,1.2],[.5,.5]))
    fit = require_converged(solve(model,StateGrid(model;k=(n=60,bounds=(.01,20.)),z=(n=4,bounds=(.8,1.2)))))
    for relative in ("counterfactuals/single-dim-kkt.jl","counterfactuals/apply-single-dim-kkt.jl")
        m = notebook_functions(relative, (:create_function_interpolation,:uniform_μ))
        value = notebook_call(m,:create_function_interpolation,fit,:value)
        forward = notebook_call(m,:create_function_interpolation,fit,:forward)
        backward = notebook_call(m,:create_function_interpolation,fit,:backward)
        k = fit.grid.k
        @test value(k[20]) ≈ fit.value[20,2,1]
        @test forward(k[20]) ≈ (fit.value[21,2,1]-fit.value[20,2,1])/(k[21]-k[20])
        @test backward(k[20]) ≈ (fit.value[20,2,1]-fit.value[19,2,1])/(k[20]-k[19])
        if occursin("apply-",relative)
            normalized, scale = notebook_call(m,:create_function_interpolation,fit,:value,2)
            expected = fit.value[:,2,2]
            @test normalized(k[20]) ≈ (expected[20]-minimum(expected))/(maximum(expected)-minimum(expected))
            @test scale ≈ 1/minimum(abs.(fit.value))
        else
            weights, shaped = notebook_call(m,:uniform_μ,fit)
            @test weights == shaped && sum(weights) ≈ 1
        end
    end
end

@testset "Structural panel notebook probability migration" begin
    m = notebook_functions("structural/simulate-panel/simulate-panel.jl",
        (:solve_model, :initial_density, :create_policy_df, :sample_path,
         :find_closest_idx, :create_init_density, :dens_vec_to_grid,
         :generate_stationary_distribution, :sample_init_indices))
    fit = notebook_call(m, :solve_model, StochasticSkibaAbilityModel();
        Nk=60,Nz=5,Nη=3,z_bounds=(.5,1.5))
    @test fit.converged
    table = notebook_call(m, :create_policy_df, fit)
    @test size(table,1) == length(fit.value)
    @test table.c == vec(fit.policy.consumption)
    one, index = notebook_call(m, :sample_path, fit, 20,3,2,1)
    @test index == CartesianIndex(20,3,2) # historical T=1 means time zero
    @test only(one.k) == fit.grid.k[20]
    @test only(one.c) == fit.policy.consumption[20,3,2]
    evolved = notebook_call(m, :generate_stationary_distribution, fit, 42)
    @test sum(mass(evolved)) ≈ 1
    @test minimum(mass(evolved)) >= 0
    indices = notebook_call(m, :sample_init_indices, size(fit.grid), evolved, 100, 42)
    @test length(indices) == 100
    @test all(i -> checkbounds(Bool,fit.value,i),indices)
end

@testset "Migrated notebook grids and predictions" begin
    for relative in ("counterfactuals/counterfactuals.jl",
            "counterfactuals/kkt-counterfactuals.jl",
            "counterfactuals/metastability.jl", "metastability/phase-plot.jl",
            "metastability/Makie.jl", "descriptives/predicted-tes.jl")
        m = notebook_functions(relative, (:setup_simple_model, :setup_productivity_model,
            :median_indices, :generate_predictions))
        fit = notebook_call(m, :setup_simple_model; Nk=40, Nz=4, Nη=3)
        @test fit.converged && size(fit.grid) == (40,4,3)
        hi = relative in ("counterfactuals/metastability.jl", "metastability/phase-plot.jl",
            "metastability/Makie.jl") ? 15. : 20.
        t = collect(range(0., 1.; length=40))
        @test fit.grid.k ≈ .001 .+ (hi-.001) .* (t .+ 5 .* t.^10) ./ 6
        @test fit.grid.ability == [.5,1.,1.5]
        if relative == "counterfactuals/metastability.jl"
            second = notebook_call(m, :setup_productivity_model; Nk=40, Nz=4)
            @test second.converged && size(second.grid) == (40,4)
        elseif relative == "descriptives/predicted-tes.jl"
            prediction = notebook_call(m, :generate_predictions, fit; single_T=true)
            @test size(prediction,1) == 40
            @test all(isfinite, prediction.c_te)
            @test prediction.kdot_control == fit.policy.drift[:,2,1]
        end
    end
    m = notebook_functions("growth-model/growth-model-results.jl", (:solve_model,))
    fit = notebook_call(m, :solve_model, StochasticSkibaAbilityModel(),
        (.01,20.), (.5,1.5), (.5,1.5))
    @test fit.converged && size(fit.grid) == (100,10,10)
    @test first(fit.grid.k) == .001
end
