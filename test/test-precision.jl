using GrowthModels
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
    hyperparams = StateSpaceHyperParams(m_skiba)
    init_value = Value(Float64, hyperparams);
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


function calculate_variance(array)
    n = size(array, 2)
    mean_array = sum(array, dims = 2) / n
    variance_array = sum((array .- mean_array).^2, dims = 2) / (n - 1)
    return variance_array[:]
end

@testset "Testing value function same across fits" begin
    value_fits, variable_fits = many_fits(test_params);

    value_variance = calculate_variance(value_fits)
    variable_variance = calculate_variance(variable_fits)

    @test maximum(value_variance) <= 1e-10
    @test maximum(variable_variance) <= 1e10
end




