using Test, GrowthModels, SparseArrays
@testset "Pre-redesign explicit-grid numerical agreement" begin
    data=Dict{Symbol,Vector{NTuple{3,Float64}}}()
    for line in readlines(joinpath(@__DIR__,"test-data","api-baseline.csv"))[2:end]
        fields=split(line,',')
        push!(get!(data,Symbol(fields[1]),NTuple{3,Float64}[]),Tuple(parse.(Float64,fields[3:5])))
    end
    p=PoissonProcess(z=[0.,.1],Q=[-.2 .2;.3 -.3])
    models=(rck=RamseyCassKoopmansModel(),skiba=SkibaModel(),smooth=SmoothSkibaModel(),
        ou=StochasticRamseyCassKoopmansModel(),poisson=StochasticSkibaModel(p),
        ability=ResourceModel(StochasticSkibaAbilityModel();population=Population([.8,1.2],[.5,.5])))
    for (name,m) in pairs(models)
        g=if name in (:rck,:skiba,:smooth)
            StateGrid(m;k=(n=60,bounds=(.01,5.)))
        elseif name==:poisson
            StateGrid(m;k=(n=60,bounds=(.01,5.)))
        else
            StateGrid(m;k=(n=60,bounds=(.01,5.)),z=(n=6,bounds=(.8,1.2)))
        end
        fit=solve(m,g)
        @test fit.converged
        reference=data[name]
        @test maximum(abs.(vec(fit.value)-first.(reference))) < 1e-7
        @test maximum(abs.(vec(fit.policy.consumption)-getindex.(reference,2))) < 1e-7
        d=distribution(g;masses=fill(1/length(g),size(g)))
        result=advance(Evolution(fit;dt=.25),d,1.)
        @test maximum(abs.(vec(mass(result))-last.(reference))) < 1e-8
    end
end
