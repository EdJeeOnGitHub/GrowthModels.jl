using GrowthModels
using Statistics
using Test

test_params = [
    2.0,
    0.3,
    0.05,
    0.05, 
    0.6,
    0.4,
    2.0
]

function wrapper(p)
    m_skiba = SkibaModel(p...)
    hyperparams = HyperParams(m_skiba)
    init_value = Value(hyperparams);
    res = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    return res
end

function many_fits(p; n = 10)
    value_array = Array{Float64}(undef, 1000, n)
    variable_array = Array{Float64}(undef, 1000, n)
    for i in 1:n
        res = wrapper(p)
        value_array[:, i] = res.value.v[:]
        variable_array[:, i] = res.variables.c[:]
    end
    return value_array, variable_array
end


@testset "Testing value function same across fits" begin
    value_fits, variable_fits = many_fits(test_params);

    @test maximum(var(value_fits, dims = 2)[:]) <= 1e-10
    @test maximum(var(variable_fits, dims = 2)[:]) <= 1e10
end




