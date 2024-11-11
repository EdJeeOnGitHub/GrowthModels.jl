using GrowthModels
using BenchmarkTools

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


p_pf = [m_skiba.α, m_skiba.A_H, m_skiba.A_L, m_skiba.κ]

k = abs.(randn(1000))

# @benchmark production_function($m_skiba, $k)
# @benchmark production_function($m_skiba, $k, $p_pf)
# @benchmark skiba_production_function($k[1], $m_skiba.α, $m_skiba.A_H, $m_skiba.A_L, $m_skiba.κ)
# @benchmark production_function($m_skiba, $k, $m_skiba.α, $m_skiba.A_H, $m_skiba.A_L, $m_skiba.κ)


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



suite["skiba"]["production_function"]["full_pass"] = @benchmarkable production_function($m_skiba, 1.0, $m_skiba.α, $m_skiba.A_H, $m_skiba.A_L, $m_skiba.κ)
suite["skiba"]["production_function"]["param_pass"] = @benchmarkable skiba_production_function(1.0, $m_skiba.α, $m_skiba.A_H, $m_skiba.A_L, $m_skiba.κ)
suite["skiba"]["production_function"]["struct_pass"] = @benchmarkable production_function($m_skiba, 1.0)
suite["skiba"]["kdot_function"] = @benchmarkable r_skiba.kdot_function(1.2);

kdot_bench = @benchmark r_skiba.kdot_function(1.0);

tune!(suite)

results = run(suite, verbose = true)


BenchmarkTools.save("output.json", median(results))
