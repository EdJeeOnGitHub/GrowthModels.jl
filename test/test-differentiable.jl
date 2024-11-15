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

function policy_wrapper_inplace(p; x = 1.0, init_value = Value(Real, (1000,)))
    m_skiba = SkibaModel(p...)
    hyperparams = StateSpaceHyperParams(m_skiba)
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    # Updating In Place
    init_value.v[:] = ForwardDiff.value.(fit_value.v)
    init_value.dVf[:] = ForwardDiff.value.(fit_value.dVf)
    init_value.dVb[:] = ForwardDiff.value.(fit_value.dVb)
    init_value.dV0[:] = ForwardDiff.value.(fit_value.dV0)
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    return r_skiba.policy_function(x...)
end



function policy_wrapper(p; x = 1.0)
    m_skiba = SkibaModel(p...)
    hyperparams = StateSpaceHyperParams(m_skiba)
    init_value = Value(Real, hyperparams);
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    return r_skiba.policy_function(x...)
end

policy_wrapper(test_params)

# @testset "Policy function differentiable wrt params" begin
#     init_value = Value(Real, (1000,) )
#     g = ForwardDiff.gradient(p -> policy_wrapper(p, x = 1.0), test_params);
#     @test isa(g, Vector);
#     g
#     for i in eachindex(g)
#         @test isnan(g[i]) == false
#     end
#     inplace_value = Value(Real, (1000,))
#     g_inplace_1 = ForwardDiff.gradient(x -> policy_wrapper_inplace(x, init_value = inplace_value), test_params)
#     g_inplace_2 = ForwardDiff.gradient(x -> policy_wrapper_inplace(x, init_value = inplace_value), test_params)
#     g_inplace_3 = ForwardDiff.gradient(x -> policy_wrapper_inplace(x, init_value = inplace_value), test_params)
#     @test isa(g_inplace_1, Vector);
#     for i in eachindex(g_inplace_1)
#         @test isnan(g_inplace_1[i]) == false
#         # first and second can be off due to first value function being far from 
#         # true value but second -> are very close to true value
#         @test abs(g_inplace_2[i] - g_inplace_3[i]) < 1e-8
#     end
# end


function kdot_wrapper(p; x = 1.0)
    m_skiba = SkibaModel(p...)
    hyperparams = StateSpaceHyperParams(m_skiba)
    init_value = Value(Real, hyperparams);
    fit_value, fit_variables, fit_iter = solve_HJB(m_skiba, hyperparams, init_value = init_value, maxit = 1000);
    r_skiba = SolvedModel(m_skiba, fit_value, fit_variables)
    return r_skiba.kdot_function(x...)
end



# @testset "Kdot function differentiable wrt params" begin
#     g = ForwardDiff.gradient(kdot_wrapper, test_params);
#     @test isa(g, Vector);
#     for i in eachindex(g)
#         @test isnan(g[i]) == false
#     end
# end
