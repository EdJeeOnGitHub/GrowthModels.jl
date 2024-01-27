
struct SolvedModel{T<:Model}
    convergence_status::Bool
    state::Vector{Symbol}
    control::Vector{Symbol}
    variables::NamedTuple
    production_function::Function
    policy_function::Function
    kdot_function::Function
    m::T
end

#### Model Specific Dispatch ####
# Can probably dispatch on just model here but will see in future
function SolvedModel(m::T, value::Value, variables::NamedTuple) where T <: SkibaModel
    c_policy_function = cubic_spline_interpolation(
        variables.k,
        variables.c,
        extrapolation_bc = Interpolations.Line()
    );
    kdot_function = k -> production_function(m)(k) - m.δ*k - c_policy_function(k)    
    SolvedModel(
        value.convergence_status,
        [:k],
        [:c],
        variables,
        production_function(m),
        x -> c_policy_function(x),
        kdot_function,
        m
    )
end

function SolvedModel(m::T, res::NamedTuple) where T <: Model
    SolvedModel(m, res.value, res.variables)
end


function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, span::Tuple)
    f(u, p, t) = r.kdot_function.(u)
    k0 = state
    prob = ODEProblem(f, k0, span)
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
    return sol
end


function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, time_period::Int)
    if time_period == 0
        return state
    end
    f(u, p, t) = r.kdot_function.(u)
    k0 = state
    prob = ODEProblem(f, k0, time_period)
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8, save_everystep = false)
    return sol
end

function f_ode(model_type, solved_model)
    if model_type != "inplace"
        return (u, p, t) -> [solved_model.kdot_function(u[1])]
    else
        function (du, u, p, t)
            du[1] = solved_model.kdot_function(u[1])
            return nothing
        end
    end
end

function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, max_time::Int, timesteps::Vector, ensemble::Union{EnsembleThreads,EnsembleSerial})
    f = f_ode("inplace", r)
    k0 = state
    prob = ODEProblem(f, k0[1], max_time)
    function prob_fun(prob, i, repeat)
        remake(prob, u0 = [k0[i]])
    end
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_fun)
    sol = solve(
        ensemble_prob,
        Tsit5(),
        ensemble,
        saveat = timesteps,
        trajectories = length(k0)
    )
    return sol
end

function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, span::Tuple, ensemble::Union{EnsembleThreads,EnsembleSerial})
    f = f_ode("inplace", r)
    k0 = state
    prob = ODEProblem(f, k0[1], span)
    function prob_fun(prob, i, repeat)
        remake(prob, u0 = [k0[i]])
    end
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_fun)
    sol = solve(
        ensemble_prob,
        Tsit5(),
        ensemble,
        saveat = 1.0,
        trajectories = length(k0)
    )
    return sol
end

function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, time_period::Int, ensemble::Union{EnsembleThreads,EnsembleSerial})
    if time_period == 0
        return state
    end
    f = f_ode("inplace", r)
    
    k0 = state
    prob = ODEProblem(f, k0[1], time_period)
    function prob_fun(prob, i, repeat)
        remake(prob, u0 = [k0[i]])
    end
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_fun)
    sol = solve(
        ensemble_prob,
        Tsit5(),
        ensemble,
        save_everystep = false,
        trajectories = length(k0)
    )
    return sol
end



# Plot evolution of outcomes using ODE result and model solution
function plot_timepath(ode_result::ODESolution, r::SolvedModel{T}; N = size(ode_result, 1)) where T <: SkibaModel
    kt_path = Array(ode_result)
    ct_path = r.policy_function.(kt_path)
    yt_path = r.production_function.(kt_path)
    st_path = r.kdot_function.(kt_path)
    time = ode_result.t

    p_c = plot()
    p_k = plot()
    p_y = plot()
    p_s = plot()
    for id in 1:N
        plot!(p_k, time, kt_path[id, :], color = :blue, label = "")
        plot!(p_c, time, ct_path[id, :], color = :red, label = "")
        plot!(p_y, time, yt_path[id, :], color = :green, label = "")
    end
    plot!(p_s, kt_path[:], st_path[:], colour = :black, label = "", seriestype = :scatter)
    xlabel!(p_s, "\$k(t)\$")
    ylabel!(p_s, "\$\\dot{k}(t)\$")
    ylabel!(p_c, "\$c(t)\$")
    ylabel!(p_k, "\$k(t)\$")
    ylabel!(p_y, "\$y(t)\$")
    xlabel!(p_y, "\$t\$")
    xlabel!(p_c, "\$t\$")
    xlabel!(p_k, "\$t\$")
    p_all = plot(
        p_c, p_k, p_y, p_s
    )
    return p_all
end


function plot_timepath(ode_result::EnsembleSolution, r::SolvedModel{T}; N = size(ode_result, 3)) where T <: SkibaModel
    kt_path = Array(ode_result)
    ct_path = r.policy_function.(kt_path)
    yt_path = r.production_function.(kt_path)
    st_path = r.kdot_function.(kt_path)
    time = ode_result[1].t

    p_c = plot()
    p_k = plot()
    p_y = plot()
    p_s = plot()
    for id in 1:N
        plot!(p_k, time, kt_path[1, :, id], color = :blue, label = "")
        plot!(p_c, time, ct_path[1, :, id], color = :red, label = "")
        plot!(p_y, time, yt_path[1, :, id], color = :green, label = "")
    end
    plot!(p_s, kt_path[:], st_path[:], colour = :black, label = "", seriestype = :scatter)
    xlabel!(p_s, "\$k(t)\$")
    ylabel!(p_s, "\$\\dot{k}(t)\$")
    ylabel!(p_c, "\$c(t)\$")
    ylabel!(p_k, "\$k(t)\$")
    ylabel!(p_y, "\$y(t)\$")
    xlabel!(p_y, "\$t\$")
    xlabel!(p_c, "\$t\$")
    xlabel!(p_k, "\$t\$")
    p_all = plot(
        p_c, p_k, p_y, p_s
    )
    return p_all
end

function show(io::IO, r::SolvedModel)
    print(
        io,
        lineplot(
            r.variables.k,
            r.variables.c,
            xlabel = "k(t)",
            ylabel = "c(t)"
        )
    )
end