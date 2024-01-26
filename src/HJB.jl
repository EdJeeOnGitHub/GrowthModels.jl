
function solve_HJB(m::Model, hyperparams::HyperParams, state::StateSpace; init_value = Value(hyperparams), maxit = 1000)
    curr_iter = 0
    val = deepcopy(init_value)
    for n in 1:maxit
        curr_iter += 1
        output_value, curr_iter = update_v(m, val, state, hyperparams, iter = n)
        if output_value.convergence_status
            fit_value, _, fit_variables = update_v(m, val, state, hyperparams, iter = curr_iter, silent = true)
            return (value = fit_value, variables = fit_variables, iter = curr_iter)
            break
        end
        val = output_value
    end
    return (value = val, variables = nothing, iter = curr_iter)
end

function solve_HJB(m::Model, hyperparams::HyperParams; init_value = Value(hyperparams), maxit = 1000)
    state = StateSpace(m, hyperparams)
    return solve_HJB(m, hyperparams, state; init_value = init_value, maxit = maxit)
end