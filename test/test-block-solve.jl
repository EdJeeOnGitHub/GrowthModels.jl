using Test, GrowthModels, SparseArrays, LinearAlgebra
const BlockGM = GrowthModels

# Fixed abilities that can also switch: the operator links them, but the model is
# otherwise identical. This is the change that must not silently break the solver.
struct SwitchingAbilityModel{M} <: BlockGM.Model{Float64}
    model::M
    rate::Float64
end
BlockGM.process(m::SwitchingAbilityModel) = BlockGM.process(m.model)
BlockGM.population(m::SwitchingAbilityModel) = BlockGM.population(m.model)
BlockGM.preferences(m::SwitchingAbilityModel) = BlockGM.preferences(m.model)
BlockGM.resources(m::SwitchingAbilityModel,s::NamedTuple) = BlockGM.resources(m.model,s)
function BlockGM.exogenous_generator(m::SwitchingAbilityModel,g::BlockGM.StateGrid)
    Na = size(g)[3]
    switching = m.rate .* (ones(Na,Na) ./ Na - I)
    independent = BlockGM.exogenous_generator(m.model,g)
    return independent + kron(sparse(switching),sparse(I,prod(size(g)[1:2]),prod(size(g)[1:2])))
end

# The same solve, forced through one block. This is the reference for equivalence.
function single_block_solve(m,g;tolerance)
    ws = BlockGM.HJBWorkspace(m,g)
    start = similar(ws.next_v)
    start .= BlockGM.initial_guess(m,ws.y)
    settings = (value_tolerance=tolerance,residual_tolerance=tolerance,maxiters=1000,
        Delta=1000.,initialization=:default,algorithm=:implicit_upwind_hjb,history=false)
    return BlockGM.solve_blocks(m,g,ws,start,false,settings,[1:length(g)])
end

@testset "Operator blocks" begin
    A = sparse([1,2,3,4,5,6],[2,1,4,3,6,5],ones(6),6,6)
    @test BlockGM.operator_blocks(A) == [1:2,3:4,5:6]
    # One direction is enough to link states, and only the linked blocks merge.
    B = copy(A); B[2,3] = 1.
    @test BlockGM.operator_blocks(B) == [1:4,5:6]
    # A stored zero may become a rate later, so it also links states.
    C = copy(A); C[4,5] = 1.; C.nzval[C.nzval .== 1] .= 0.
    @test BlockGM.operator_blocks(C) == [1:2,3:6]
    # Separable but interleaved states merge into a valid contiguous block.
    D = sparse([1,3,2,4],[3,1,4,2],ones(4),4,4)
    @test BlockGM.operator_blocks(D) == [1:4]
    @test BlockGM.operator_blocks(spzeros(3,3)) == [1:1,2:2,3:3]
end

@testset "Warm-start levels" begin
    for n in (1,2,3,5,8,40)
        levels = BlockGM.warm_start_levels(n)
        seen = Int[]
        for level in levels, (block,parent) in level
            @test parent == 0 || parent in seen
            push!(seen,block)
        end
        @test sort(seen) == 1:n
        @test length(first(levels)) == 1
    end
end

@testset "Block workspaces match the joint workspace" begin
    pop = Population([.8,1.,1.2],[.3,.4,.3])
    m = ResourceModel(StochasticSkibaAbilityModel();population=pop)
    g = StateGrid(m;k=(n=20,),z=(n=4,))
    ws = BlockGM.HJBWorkspace(m,g)
    v = BlockGM.initial_guess(m,ws.y)
    blocks = BlockGM.operator_blocks(ws.A)
    @test blocks == [1:80,81:160,161:240]
    parts = [BlockGM.block_workspace(ws,r) for r in blocks]
    @test sparse(ws.A[blocks[2],blocks[2]]) == parts[2].A
    BlockGM.prepare_hjb_step!(ws,m,v,1000.)
    for (r,block) in zip(blocks,parts)
        BlockGM.prepare_hjb_step!(block,m,reshape(v[r],20,:),1000.)
        @test block.B == ws.B[r,r]
        @test block.rhs == ws.rhs[r]
        @test vec(block.c) == ws.c[r]
    end
    @test_throws ArgumentError BlockGM.block_workspace(ws,1:30)
end

@testset "Blocks agree with a single-block solve" begin
    pop = Population([.8,1.,1.2],[.3,.4,.3])
    poisson = PoissonProcess(z=[0.,.1],Q=[-.2 .2; .3 -.3])
    cases = (
        skiba=(ResourceModel(StochasticSkibaAbilityModel();population=pop),(k=(n=40,),z=(n=5,))),
        floor=(ResourceModel(StochasticSkibaAbilityModel();transfer=.1,income_floor=.3,population=pop),(k=(n=40,),z=(n=5,))),
        poisson=(ResourceModel(StochasticSkibaAbilityModel(poisson);population=pop),(k=(n=40,),)),
        functional=(FunctionalModel((k,z,a) -> a*z*k^.3,OrnsteinUhlenbeckProcess(θ=.1,σ=.1),pop),(k=(n=40,bounds=(.1,20.)),z=(n=5,))),
    )
    for (name,(m,kw)) in pairs(cases)
        @testset "$name" begin
            g = StateGrid(m;kw...)
            joint = single_block_solve(m,g;tolerance=1e-10)
            blocked = solve(m,g;value_tolerance=1e-10,residual_tolerance=1e-10)
            @test joint.converged && blocked.converged
            @test length(joint.diagnostics.blocks.iterations) == 1
            @test length(blocked.diagnostics.blocks.iterations) == size(g)[3]
            @test maximum(abs.(blocked.value .- joint.value)) < 1e-8
            @test maximum(abs.(blocked.policy.consumption .- joint.policy.consumption)) < 1e-8
            @test findnz(sparse(blocked.generator))[1:2] == findnz(sparse(joint.generator))[1:2]
        end
    end
end

@testset "Models without separable structure are one block" begin
    pop = Population([.8,1.2],[.5,.5])
    base = ResourceModel(StochasticSkibaAbilityModel();population=pop)
    g = StateGrid(base;k=(n=40,),z=(n=5,))
    fit = solve(SwitchingAbilityModel(base,.1),g)
    @test fit.converged
    @test length(fit.diagnostics.blocks.iterations) == 1
    # Switching actually reaches the solution: types are pulled towards each other.
    independent = solve(base,g)
    @test maximum(abs.(fit.value .- independent.value)) > 1e-2
    @test sum(abs,fit.value[:,:,1] .- fit.value[:,:,2]) < sum(abs,independent.value[:,:,1] .- independent.value[:,:,2])
end

@testset "Diagnostics and reproducibility" begin
    pop = Population([.8,1.,1.2],[.3,.4,.3])
    m = ResourceModel(StochasticSkibaAbilityModel();population=pop)
    g = StateGrid(m;k=(n=40,),z=(n=5,))
    fit = solve(m,g;history=true)
    blocks = fit.diagnostics.blocks
    @test blocks.iterations isa AbstractVector && length(blocks.iterations) == 3
    @test all(==(:converged),blocks.stop_reason)
    @test fit.diagnostics.iterations == maximum(blocks.iterations)
    @test length(fit.diagnostics.history) == fit.diagnostics.iterations
    @test last(fit.diagnostics.history).residual == fit.diagnostics.residual_norm
    # Warm starts are visible: the root solves in full and its neighbours do not.
    @test blocks.iterations[2] > maximum(blocks.iterations[[1,3]])
    # fit.initial_value is what actually ran, so solving from it replays the iterates.
    replay = solve(m,g;initial=fit.initial_value,Delta=fit.settings.Delta)
    @test replay.value == fit.value
    @test replay.diagnostics.blocks.iterations == blocks.iterations
    limited = solve(m,g;maxiters=0)
    @test !limited.converged && all(==(0),limited.diagnostics.blocks.iterations)
    @test limited.diagnostics.stop_reason == :iteration_limit
    @test_throws ArgumentError solve(m,g;initial=fill(NaN,size(g)))
    # A caller's guess is never silently replaced by a cold start.
    @test_throws ArgumentError solve(m,g;initial=fill(Inf,size(g)))
end

@testset "Results do not depend on the thread count" begin
    script = joinpath(@__DIR__,"test-block-threads.jl")
    results = map((1,4)) do threads
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t $threads $script`
        Dict(split(line,"=";limit=2) for line in eachline(command))
    end
    single,threaded = results
    @test single["value"] == threaded["value"]
    @test single["warm"] == threaded["warm"]
    for result in results
        @test result["nested_dynamic"] == "true"
        @test result["nested_static"] == "true"
        @test result["nested_gradient"] == "true"
        @test result["custom_converged"] == "true"
        @test result["custom_off_caller"] == "0"
        @test result["error_type"] == "ArgumentError"
        @test result["error_message"] == "block 2"
    end
end
