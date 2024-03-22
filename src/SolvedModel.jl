
function plot_production_function(m::StochasticModel, k, z)
    y = production_function(m, collect(k), collect(z)')
    plot(k, y, label="")
    xlabel!("\$k\$")
    ylabel!("\$f(k)\$")
end

function plot_model(m::StochasticModel, value::Value, variables::NamedTuple)
    (; k, z, y, c) = variables
    (; v, dVf, dVb, dV0, dist) = value
    kstar = k_star(m)
    fit_kdot = GrowthModels.statespace_k_dot(m)(variables)

    p1 =  plot_production_function(m, k[:, 1], z[1, :])
    scatter!(p1, [kstar], [production_function(m, kstar, sum(z[:])/length(z[:]) )], label="kstar", markersize=4)

    index = findmin(abs.(kstar .- k))[2]
    p2 = plot(k, v, label="V")
    plot!(legend = false)
    scatter!(p2, [kstar], [v[index]], label="kstar", markersize=4)
    xlabel!(p2, "\$k\$")
    ylabel!(p2, "\$v(k)\$")

    p3 = plot(k, c, label="")
    plot!(p3, k, y .- m.δ .* k, label="", linestyle = :dash)
    xlabel!(p3, "\$k\$")
    ylabel!(p3, "\$c(k)\$")

    p4 = plot(k, fit_kdot, label="")
    # scatter!(p4, [kstar], [0], label="kstar", markersize=4)
    hline!(p4, [0], linestyle = :dash, label="kdot = 0")
    xlabel!(p4, "\$k\$")
    ylabel!(p4, "\$s(k)\$")

    subplot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))

    return subplot
end

function plot_model(m::DeterministicModel, value::Value, variables::NamedTuple)
    (; k, y, c) = variables
    (; v, dVf, dVb, dV0, dist) = value
    kstar = k_star(m)
    fit_kdot = statespace_k_dot(m)(variables)

    # subplot = plot(layout = (2, 2), size = (800, 600))
    p1 =  plot_production_function(m, collect(k))
    scatter!(p1, [kstar], [production_function(m, kstar)], label="kstar", markersize=4)

    index = findmin(abs.(kstar .- k))[2]
    p2 = plot(k, v, label="V")
    scatter!(p2, [kstar], [v[index]], label="kstar", markersize=4)
    xlabel!(p2, "\$k\$")
    ylabel!(p2, "\$v(k)\$")

    p3 = plot(k, c, label="Consumption, c(k)")
    plot!(p3, k, y .- m.δ .* k, label="Production net of depreciation, f(k) - δk")
    xlabel!(p3, "\$k\$")
    ylabel!(p3, "\$c(k)\$")

    p4 = plot(k, fit_kdot, label="kdot")
    plot!(p4, k, zeros(length(k)), linestyle=:dash, label="zeros")
    scatter!(p4, [kstar], [0], label="kstar", markersize=4)
    xlabel!(p4, "\$k\$")
    ylabel!(p4, "\$s(k)\$")

    subplot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))

    return subplot
end

struct SolvedModel{T<:Model}
    convergence_status::Bool
    state::Vector{Symbol}
    control::Vector{Symbol}
    variables::NamedTuple
    production_function::Function
    production_function_prime::Function
    policy_function::Union{Function, Interpolations.Extrapolation}
    kdot_function::Function
    ydot_function::Function
    cdot_function::Function
    m::T
end

#### Model Specific Dispatch ####
# Can probably dispatch on just model here but will see in future
function SolvedModel(m::T, value::Value, variables::NamedTuple) where T <: Union{SkibaModel,SmoothSkibaModel,RamseyCassKoopmansModel}
    c_interpolation = interpolate(
        (variables.k[:, 1], ),
        variables.c,
        Gridded(Linear())
    )
    c_policy_function = extrapolate(c_interpolation, Line())
    prod_func = x -> production_function(m, x)
    prod_func_prime = x -> production_function_prime(m, x)
    kdot_function = k -> prod_func(k) - m.δ*k - c_policy_function(k)    
    ydot_function = (k, kdot) -> prod_func(k) * kdot
    ydot_function = k -> prod_func(k) * kdot_function(k)

    function cdot_function(c, k, γ, ρ, δ) 
        cdot = (c/γ) * (prod_func_prime(k) - ρ - δ)
        return cdot
    end
    function cdot_function(c, k)
        cdot_function(c, k, m.γ, m.ρ, m.δ)
    end

    SolvedModel(
        value.convergence_status,
        [:k],
        [:c],
        variables,
        prod_func,
        prod_func_prime,
        x -> c_policy_function(x),
        kdot_function,
        ydot_function,
        cdot_function,
        m
    )
end


function SolvedModel(m::StochasticModel, value::Value, variables::NamedTuple) 
    c_interpolation = interpolate(
        (variables.k[:, 1], variables.z[1, :]),
        variables.c,
        Gridded(Linear())
    )
    c_policy_function = extrapolate(c_interpolation, Line())
    prod_func = (k, z) -> production_function(m, k, z)
    prod_func_prime = (k, z) -> production_function_prime(m, k, z)
    # need to .dot the policy function or it will give a matrix (Nk x Nz) even 
    # if Nk == Nz
    kdot_function = (k, z) -> prod_func(k, z) - m.δ*k - c_policy_function.(k, z)    
    ydot_function = (k, z, kdot) -> throw("Not yet implemented - need to add z evolution")
    ydot_function = k -> throw("Not yet implemented - need to add z evolution")

    function cdot_function(c, k, γ, ρ, δ) 
        # cdot = (c/γ) * (prod_func_prime(k) - ρ - δ)
        throw("Not yet implemented - need to add z evolution")
        return cdot
    end
    function cdot_function(c, k)
        throw("Not yet implemented - need to add z evolution")
        # cdot_function(c, k, m.γ, m.ρ, m.δ)
    end

    SolvedModel(
        value.convergence_status,
        [:k, :z],
        [:c],
        variables,
        prod_func,
        prod_func_prime,
        c_policy_function,
        kdot_function,
        ydot_function,
        cdot_function,
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

function(r::SolvedModel)(state_dict::Dict, ensemble; algorithm = Tsit5())
    function f(du, u, p, t)
        du[1] = r.kdot_function(u[1])
        return nothing
    end

    flat_states = vcat(values(state_dict)...)
    flat_keys = vcat([repeat([key], length(value)) for (key, value) in state_dict]...)
    n_indiv = length(flat_keys)

    max_time = maximum(flat_keys)
    prob = ODEProblem(f, [flat_states[1]], max_time, save_everystep = false)

    function prob_fun(prob, i, repeat)
            remake(prob, u0 = [flat_states[i]], tspan = flat_keys[i], save_everystep = false)
    end

    ensemble_prob = EnsembleProblem(
        prob, 
        prob_func = prob_fun
    )
    sol = solve(
        ensemble_prob,
        algorithm,
        ensemble,
        trajectories = n_indiv 
    )
    return sol
end

function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, max_time::Int, timesteps::Vector, ensemble)
    function f(du, u, p, t)
        du[1] = r.kdot_function(u[1])
        return nothing
    end
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

function(r::SolvedModel)(state::Union{Number,Vector{<:Number}}, time_span::Tuple, ensemble)
    function f(du, u, p, t)
        du[1] = r.kdot_function(u[1])
        return nothing
    end
    k0 = state
    prob = ODEProblem(f, k0[1], time_span)
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

function(r::SolvedModel)(state_dict::Dict, ensemble::DiffEqBase.EnsembleAlgorithm; algorithm::Union{DiffEqBase.AbstractDEAlgorithm,Nothing} = Tsit5(), reltol = 1e-6, abstol = 1e-6)
    function f(du, u, p, t)
        du[1] = r.kdot_function(u[1])
        return nothing
    end

    flat_states = vcat(values(state_dict)...)
    flat_keys = vcat([repeat([key], length(value)) for (key, value) in state_dict]...)
    n_indiv = length(flat_keys)

    max_time = maximum(flat_keys)
    prob = ODEProblem(f, flat_states[1], max_time, save_everystep = false)

    function prob_fun(prob, i, repeat)
            remake(prob, u0 = [flat_states[i]], tspan = flat_keys[i], save_everystep = false)
    end

    ensemble_prob = EnsembleProblem(
        prob, 
        prob_func = prob_fun
    )
    sol = solve(
        ensemble_prob,
        algorithm,
        ensemble,
        reltol = reltol,
        abstol = abstol,
        trajectories = n_indiv 
    )
    return sol
end

function(r::SolvedModel)(k0::Real, timesteps::Vector; algorithm = AutoTsit5(Rosenbrock23()), reltol = 1e-6, abstol = 1e-6)
    function f(du, u, p, t)
        du[1] = r.kdot_function(u[1])
        return nothing
    end

    prob = ODEProblem(f, [k0], maximum(timesteps))


    sol = solve(
        prob,
        algorithm,
        reltol = reltol,
        abstol = abstol,
        saveat = timesteps,
        save_everystep = false,
        force_dtmin = true,
        maxiters = 5000,
        isoutofdomain = (m,p,t) -> any(x->x<eps(), m)
    )
    return sol
end


# Plot evolution of outcomes using ODE result and model solution
function plot_timepath(ode_result::ODESolution, r::SolvedModel{T}; N = size(ode_result, 1)) where T <: Union{SkibaModel,SmoothSkibaModel,RamseyCassKoopmansModel}
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


function plot_timepath(ode_result::EnsembleSolution, r::SolvedModel{T}; N = size(ode_result, 3)) where T <: Union{SkibaModel,SmoothSkibaModel,RamseyCassKoopmansModel}
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

function show(io::IO, r::SolvedModel{T})  where {T <: DeterministicModel}
    print(
        io,
        lineplot(
            r.variables.k[:],
            r.variables.c[:],
            xlabel = "k(t)",
            ylabel = "c(t)"
        )
    )
end

function show(io::IO, r::SolvedModel{T})  where {T <: StochasticModel}
    median_idx = round(Int, size(r.variables.k, 2) / 2)
    print(
        io,
        lineplot(
            r.variables.k[:, median_idx],
            r.variables.c[:, median_idx],
            xlabel = "k(t)",
            ylabel = "c(t)"
        )
    )
end
