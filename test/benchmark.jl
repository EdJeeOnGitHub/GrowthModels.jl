using GrowthModels
using BenchmarkTools
using OrdinaryDiffEq

test_params = [
    2.0,
    0.3,
    0.05,
    0.05, 
    0.6,
    0.4,
    2.0
]

p = test_params
m_skiba = SkibaModel(p...)
hyperparams = HyperParams(m_skiba)
init_value = Value(hyperparams);

res = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
r_skiba = SolvedModel(m_skiba, res)

data_states = Dict(
    24 => abs.(randn(1_000)) .+ 1,
    32 => abs.(randn(1_000)) .+ 1,
    48 => abs.(randn(1_000)) .+ 1
)


suite = BenchmarkGroup()
suite["skiba"] = BenchmarkGroup(["tag1", "tag2"])

suite["skiba"]["HJB"] = @benchmarkable solve_HJB($m_skiba, $hyperparams, maxit = 1000)
suite["skiba"]["ODE"] = @benchmarkable r_skiba($data_states, EnsembleSerial(), algorithm = BS3())


tune!(suite)

results = run(suite, verbose = true)


BenchmarkTools.save("output.json", median(results))
