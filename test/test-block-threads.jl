# Run by test-block-solve.jl under several thread counts. Prints one result line per
# check, because hashes must be compared across processes.
using GrowthModels, ForwardDiff
const GM = GrowthModels

pop = Population([.8,.9,1.,1.1,1.2],fill(.2,5))
base = StochasticSkibaAbilityModel()
model = ResourceModel(base;population=pop)
grid = StateGrid(model;k=(n=40,),z=(n=5,))
fit = solve(model,grid)
println("value=",hash(collect(fit.value)))
println("warm=",hash(collect(solve(model,grid;initial=fit.value).value)))

# Solves inside an outer threaded loop, including the sticky :static schedule.
function nested_dynamic()
    out = Vector{Any}(undef,4)
    Threads.@threads :dynamic for i in 1:4
        out[i] = solve(model,grid).value
    end
    return all(x -> x == fit.value,out)
end
function nested_static()
    out = Vector{Any}(undef,4)
    Threads.@threads :static for i in 1:4
        out[i] = solve(model,grid).value
    end
    return all(x -> x == fit.value,out)
end
println("nested_dynamic=",nested_dynamic())
println("nested_static=",nested_static())

# Gradients in an outer loop match a plain gradient.
gradient(x) = sum(solve(ResourceModel(base;transfer=x,population=pop),grid).value)
plain = ForwardDiff.derivative(gradient,.1)
looped = Vector{Float64}(undef,4)
Threads.@threads for i in 1:4
    looped[i] = ForwardDiff.derivative(gradient,.1)
end
println("nested_gradient=",all(==(plain),looped))

# Custom production runs only on the calling task, so it need not be thread safe.
caller = current_task()
off_caller = Threads.Atomic{Int}(0)
production(k,z,a) = (current_task() === caller || Threads.atomic_add!(off_caller,1); a*z*k^.3)
custom = FunctionalModel(production,OrnsteinUhlenbeckProcess(θ=.1,σ=.1),pop)
customfit = solve(custom,StateGrid(custom;k=(n=40,bounds=(.1,20.)),z=(n=5,)))
println("custom_converged=",customfit.converged)
println("custom_off_caller=",off_caller[])

# Worker errors surface unwrapped and in item order.
failure = try
    GM.run_blocks(i -> i >= 2 ? throw(ArgumentError("block $i")) : i,1:4)
catch err
    err
end
println("error_type=",typeof(failure))
println("error_message=",failure.msg)
