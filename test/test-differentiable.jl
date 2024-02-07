using GrowthModels
using Test
using ForwardDiff

test_params = [
    2.0,
    0.3,
    0.05,
    0.05, 
    0.6,
    0.4,
    2.0
]

function policy_wrapper(p; x = 1.0)
    m_skiba = SkibaModel(p...)
    hyperparams = HyperParams(m_skiba)
    init_value = Value(hyperparams);
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    return r_skiba.policy_function(x...)
end

@testset "Policy function differentiable wrt params" begin
    g = ForwardDiff.gradient(policy_wrapper, test_params);
    @test isa(g, Vector);
    for i in eachindex(g)
        @test isnan(g[i]) == false
    end
end


function kdot_wrapper(p; x = 1.0)
    m_skiba = SkibaModel(p...)
    hyperparams = HyperParams(m_skiba)
    init_value = Value(hyperparams);
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    return r_skiba.kdot_function(x...)
end



@testset "Kdot function differentiable wrt params" begin
    g = ForwardDiff.gradient(kdot_wrapper, test_params);
    @test isa(g, Vector);
    for i in eachindex(g)
        @test isnan(g[i]) == false
    end
end



function loss_wrapper(p; x = 0.2)
    m_skiba = SkibaModel(p...)
    hyperparams = HyperParams(m_skiba)
    init_value = Value(hyperparams);
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    sol = r_skiba(convert.(eltype(p), x), (0.0, 24.0))
    ode_path = Array(sol)
    fake_path = zeros(size(sol))
    l = sum(abs2, ode_path .- fake_path)
    return l
end


@testset "Loss function differentiable wrt params" begin
    g = ForwardDiff.gradient(loss_wrapper, test_params);
    @test isa(g, Vector);
    for i in eachindex(g)
        @test isnan(g[i]) == false
    end
end
