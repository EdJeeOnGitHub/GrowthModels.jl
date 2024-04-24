using GrowthModels
using Plots
using ForwardDiff
using FiniteDifferences
using ColorSchemes
using TensorOperations


function solve_growth_model_fixed_grid(m::StochasticSkibaModel{T, S}) where {T <: Real, S <: StochasticProcess}
    p = m.stochasticprocess
    k_hps = HyperParams(N = 50, xmax = T(7.5), xmin = T(1e-3))
    z_hps = HyperParams(N = 2, xmax = T(maximum(p.z)), xmin = T(minimum(p.z)))
    hyperparams = StateSpaceHyperParams((k = k_hps, z = z_hps))

    state = StateSpace(m, hyperparams)
    init_value = Value(state)
    res = solve_HJB(m, hyperparams, init_value = init_value, maxit = 1000, verbose = false)

    sm = SolvedModel(m, res)
    return sm, res
end

function pack_model_params(::Type{StochasticSkibaModel}, ::Type{PoissonProcess}, param_vector)
    # first 4 params used for stochastic process
    param_tuple = (γ = eltype(param_vector)(3.0), α = param_vector[5], ρ = eltype(param_vector)(0.1), 
     δ = param_vector[6], A_H = param_vector[7] + param_vector[8], A_L = param_vector[8], 
     κ = param_vector[9]
    )
    return param_tuple
end

function pack_process_params(::Type{PoissonProcess}, param_vector)
    return (z = param_vector[1:2], λ = param_vector[3:4])
end

function instantiate_model(model_type::Type{M}, process_type::Type{S}, param_vector) where {M <: Model, S <: StochasticProcess}
    process_params = pack_process_params(process_type, param_vector)
    model_params = pack_model_params(model_type, process_type, param_vector)
    process = process_type(process_params...)
    m = model_type(
        process;
        model_params...
        )
    return m
end



function wrapper_function(param_vector)
    m = instantiate_model(StochasticSkibaModel, PoissonProcess, param_vector)
    sm, res = solve_growth_model_fixed_grid(m)
    return sm, res
end

function policy_wrapper_function(param_vector)
    sm, res = wrapper_function(param_vector)
    return sm.variables[:c][:, 1]
end

function policy_wrapper_function!(output, param_vector)
    output .= wrapper_function(param_vector)[1].variables[:c][:, 1]
end

test_param_vec = [
    -0.25,
    0.25,
    9/10,
    1/10,
    # 4.0, # γ
    0.3, # α
    # 0.05, # ρ
    0.05, # δ
    0.5, # 0.2 + A_L
    0.05, # A_L
    4.5 # κ
]

sm, res = wrapper_function(test_param_vec);
policy_matrix = policy_wrapper_function(test_param_vec)
using ForwardDiff
policy_jac = ForwardDiff.jacobian(policy_wrapper_function, test_param_vec);


function vector_hessian(f, x, out_size)
    n = size(x, 1)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
    return reshape(out, out_size, n, n)
end




function fwd_taylor_approx(f, approx_params)
        f_p = f(approx_params)

        df = ForwardDiff.jacobian(f, approx_params);
        hessian = vector_hessian(f, approx_params, size(df, 1))
    function(x)
        x_diff = x - approx_params
        array_x_diff = reshape(x_diff, 1, 1, size(x, 1))
        hess_collapse_1 =  sum(hessian .* array_x_diff, dims = 3)[:, :, 1]
        hess_diff_vec = sum(x_diff' .* hess_collapse_1, dims = 2)[:, 1]
        return f_p + df * x_diff + 0.5 * hess_diff_vec
    end
end


function third_taylor_approx(f, approx_params)
    # Evaluate function at the given parameters
    f_p = f(approx_params)
    # Set up finite difference methods
    diff_method = central_fdm(3, 1)
    diff_method_no_adapt = central_fdm(3, 1, adapt = 0)

    # First and second derivatives
    df = jacobian(diff_method, f, approx_params)[1]
    hessian = jacobian(diff_method, x -> jacobian(diff_method_no_adapt, f, x)[1], approx_params)[1]
    hessian = reshape(hessian, (size(df, 1), size(approx_params, 1), size(approx_params, 1)))

    # Third derivatives (using nested jacobians)
    third_derivatives = jacobian(diff_method, x -> jacobian(diff_method_no_adapt, y -> jacobian(diff_method_no_adapt, f, y)[1], x), approx_params)[1]
    third_derivatives = reshape(third_derivatives, (size(df, 1), size(approx_params, 1), size(approx_params, 1), size(approx_params, 1)));

    # Define function for Taylor approximation with third-order term
    function(x)
        x_diff = x - approx_params
        # Calculate first-order term
        first_order_term = df * x_diff

        # Calculate second-order term
        array_x_diff = reshape(x_diff, 1, 1, size(x, 1))
        hess_collapse_1 = sum(hessian .* array_x_diff, dims = 3)[:, :, 1]
        second_order_term = 0.5 * sum(x_diff' .* hess_collapse_1, dims = 2)[:, 1]


        @tensor begin
            third_order_term[i] := third_derivatives[i, j, k, l] * x_diff[j] * x_diff[k] * x_diff[l]
        end
        third_order_term = third_order_term / 6

        # Return the total approximation
        return f_p + first_order_term + second_order_term + third_order_term
    end
end

t_approx_2 = taylor_approx(policy_wrapper_function, test_param_vec);
t_approx_3 = third_taylor_approx(policy_wrapper_function, test_param_vec);

test_fns = (policy_wrapper_function, t_approx_2, t_approx_3)

function vary_model_parameter(new_param_vector, param_idx, full_param_vector, fns)
    true_f, t_approx_2, t_approx_3 = fns
    true_pi_stars = map(new_param_vector) do p
        temp_param_vector = copy(full_param_vector)
        temp_param_vector[param_idx] = p
        true_f(temp_param_vector)
    end
    second_order_approx_pi_stars = map(new_param_vector) do p
        temp_param_vector = copy(full_param_vector)
        temp_param_vector[param_idx] = p
        t_approx_2(temp_param_vector)
    end
    third_order_approx_pi_stars = map(new_param_vector) do p
        temp_param_vector = copy(full_param_vector)
        temp_param_vector[param_idx] = p
        t_approx_3(temp_param_vector)
    end
    differences = new_param_vector .- full_param_vector[param_idx]
    return true_pi_stars, second_order_approx_pi_stars, third_order_approx_pi_stars, differences
end



kappas = collect(range(4.25, 4.75, length = 10))
varying_kappa_results = vary_model_parameter(kappas, 9, test_param_vec, test_fns)

A_Hs = collect(range(test_param_vec[7] - 0.25, test_param_vec[7] + 0.25, length = 10))
varying_A_H_results = vary_model_parameter(A_Hs, 7, test_param_vec, test_fns)

neg_shock_size = collect(range(test_param_vec[1] - 0.25, test_param_vec[1] + 0.25, length = 10))
varying_neg_shock_size = vary_model_parameter(neg_shock_size, 1, test_param_vec, test_fns)

function plot_approximations(k, pi_true, pi_sot, pi_tot, differences; variable = "")
    normalized_differences = (differences .- minimum(differences)) / (maximum(differences) - minimum(differences))
    gradient = ColorSchemes.inferno.colors  # You can choose any other available gradient
    color = [gradient[ceil(Int, normalized_differences[i] * (length(gradient) - 1)) + 1] for i in 1:length(pi_true)]'

    p_true = plot(
        k,
        pi_true,
        ylabel = "\$c(k)\$",
        xlabel = "\$k\$",
        labels = "",
        title = "True Policy Function: Varying $variable",
        color = color,
        markerstrokewidth = 0,  # Optional: removes marker borders for a cleaner look
        legend = false  # since you specified labels=""
    )

    p_second = plot(
        k,
        pi_sot,
        ylabel = "\$c(k)\$",
        xlabel = "\$k\$",
        labels = "",
        title = "Second Order Taylor Approximation: Varying $variable",
        color = color,
        markerstrokewidth = 0,  # Optional: removes marker borders for a cleaner look
        legend = false  # since you specified labels=""
    )

    p_third = plot(
        k,
        pi_tot,
        ylabel = "\$c(k)\$",
        xlabel = "\$k\$",
        label = round.(differences, digits = 3)',
        legend = :outertopright,
        title = "Third Order Taylor Approximation: Varying $variable",
        color = color,
        markerstrokewidth = 0,  # Optional: removes marker borders for a cleaner look
    )
    p_all = plot(p_true, p_second, p_third, layout = (3, 1), link = :both)
    return p_all
end

k_single_vec = sm.variables[:k][:, 1]
plot_approximations(
    k_single_vec,
    varying_kappa_results...,
    variable = "κ"
)
savefig(
    "temp-data/kappa-taylor-approx.png"
)
plot_approximations(k_single_vec, varying_A_H_results..., variable = "A_H")
savefig(
    "temp-data/AH-taylor-approx.png"
)
plot_approximations(k_single_vec, varying_neg_shock_size..., variable = "-ve Shock")
savefig(
    "temp-data/neg-shock-taylor-approx.png"
)






