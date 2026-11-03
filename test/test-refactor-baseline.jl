# Generate only under commit 067806e, before implementation changes:
# julia --project=. test/test-refactor-baseline.jl --generate
using Test, GrowthModels, SparseArrays

function refactor_cases()
    poisson = PoissonProcess(z=[0., .1], Q=[-.2 .2; .3 -.3])
    models = (rck=RamseyCassKoopmansModel(), skiba=SkibaModel(),
        smooth=SmoothSkibaModel(), ou=StochasticRamseyCassKoopmansModel(),
        poisson=StochasticSkibaModel(poisson),
        ability=ResourceModel(StochasticSkibaAbilityModel(); population=Population([.8,1.2],[.3,.7])),
        stretched=StochasticRamseyCassKoopmansModel())
    map(collect(pairs(models))) do (name, model)
        k = name == :stretched ? (n=300,bounds=(.01,5.),spacing=:stretched,power=1.5) : (n=60,bounds=(.01,5.))
        grid = name in (:rck,:skiba,:smooth,:poisson) ? StateGrid(model;k=k) :
            StateGrid(model;k=k,z=(n=name == :stretched ? 20 : 6,bounds=(.8,1.2)))
        (name,model,grid)
    end
end

function refactor_snapshot(model, grid)
    fit = solve(model,grid)
    require_converged(fit)
    initial = distribution(grid;masses=fill(1/length(grid),size(grid)))
    p = mass(advance(Evolution(fit;dt=.25),initial,1.))
    rows,cols,rates = findnz(sparse(fit.generator))
    (value=vec(fit.value),consumption=vec(fit.policy.consumption),
     drift=vec(fit.policy.drift),resources=vec(fit.policy.resources),
     rows=rows,cols=cols,rates=rates,probability=vec(p),
     converged=[Int(fit.converged)],iterations=[fit.diagnostics.iterations])
end

fixture = joinpath(@__DIR__,"test-data","refactor-baseline.csv")
if "--generate" in ARGS
    open(fixture,"w") do io
        println(io,"case,quantity,index,value")
        for (name,model,grid) in refactor_cases()
            snapshot=refactor_snapshot(model,grid)
            for (quantity,data) in pairs(snapshot), i in eachindex(data)
                println(io,name,",",quantity,",",i,",",data[i])
            end
            println(name,": iterations=",only(snapshot.iterations))
        end
    end
    write(joinpath(@__DIR__,"test-data","refactor-baseline-provenance.txt"),
        "Source commit: 067806e\nJulia: $(VERSION)\nMachine: $(Sys.MACHINE)\n" *
        "Generator: test/test-refactor-baseline.jl --generate\n" *
        "Settings: solve defaults; evolution dt=0.25, duration=1; uniform initial mass.\n" *
        "Grids and models: refactor_cases() in generator.\n" *
        "Strict iteration comparison: GROWTHMODELS_STRICT_ITERATIONS=true in same runtime.\n")
else
    @testset "Structural refactor baseline (067806e)" begin
        expected=Dict{Tuple{Symbol,Symbol},Vector{Float64}}()
        for line in readlines(fixture)[2:end]
            name,quantity,index,value=split(line,',')
            push!(get!(expected,(Symbol(name),Symbol(quantity)),Float64[]),parse(Float64,value))
        end
        for (name,model,grid) in refactor_cases()
            snapshot=refactor_snapshot(model,grid)
            for (quantity,data) in pairs(snapshot)
                reference=expected[(name,quantity)]
                if quantity == :iterations
                    if get(ENV,"GROWTHMODELS_STRICT_ITERATIONS","false") == "true"
                        @test data == reference
                    elseif data != reference
                        @info "Iteration count differs across environments" name expected=reference actual=data
                    end
                elseif quantity in (:rows,:cols,:converged)
                    @test data == reference
                else
                    @test maximum(abs.(data .- reference)) <= 1e-11
                end
            end
        end
    end
end
