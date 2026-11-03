
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
    reference_label = m isa Union{SkibaModel,StochasticSkibaModel} ? "technology crossing" : "reference capital"
    fit_kdot = GrowthModels.statespace_k_dot(m)(variables)

    p1 =  plot_production_function(m, k[:, 1], z[1, :])
    scatter!(p1, [kstar], [production_function(m, kstar, sum(z[:])/length(z[:]) )], label=reference_label, markersize=4)

    index = findmin(abs.(kstar .- k))[2]
    p2 = plot(k, v, label="V")
    plot!(legend = false)
    scatter!(p2, [kstar], [v[index]], label=reference_label, markersize=4)
    xlabel!(p2, "\$k\$")
    ylabel!(p2, "\$v(k)\$")

    p3 = plot(k, c, label="")
    plot!(p3, k, y .- m.δ .* k, label="", linestyle = :dash)
    xlabel!(p3, "\$k\$")
    ylabel!(p3, "\$c(k)\$")

    p4 = plot(k, fit_kdot, label="")
    # scatter!(p4, [kstar], [0], label=reference_label, markersize=4)
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
    reference_label = m isa Union{SkibaModel,StochasticSkibaModel} ? "technology crossing" : "reference capital"
    fit_kdot = statespace_k_dot(m)(variables)

    # subplot = plot(layout = (2, 2), size = (800, 600))
    p1 =  plot_production_function(m, collect(k))
    scatter!(p1, [kstar], [production_function(m, kstar)], label=reference_label, markersize=4)

    index = findmin(abs.(kstar .- k))[2]
    p2 = plot(k, v, label="V")
    scatter!(p2, [kstar], [v[index]], label=reference_label, markersize=4)
    xlabel!(p2, "\$k\$")
    ylabel!(p2, "\$v(k)\$")

    p3 = plot(k, c, label="Consumption, c(k)")
    plot!(p3, k, y .- m.δ .* k, label="Production net of depreciation, f(k) - δk")
    xlabel!(p3, "\$k\$")
    ylabel!(p3, "\$c(k)\$")

    p4 = plot(k, fit_kdot, label="kdot")
    plot!(p4, k, zeros(length(k)), linestyle=:dash, label="zeros")
    scatter!(p4, [kstar], [0], label=reference_label, markersize=4)
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
    value::Value
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
    kdot_function = k -> prod_func(k) + get(variables, :flat_transfer, 0.0) - m.δ*k - c_policy_function(k)
    ydot_function = (k, kdot) -> prod_func_prime(k) * kdot
    ydot_function = k -> prod_func_prime(k) * kdot_function(k)

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
        value,
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
    kdot_function = (k, z) -> prod_func(k, z) + get(variables, :flat_transfer, 0.0) - m.δ*k - c_policy_function.(k, z)
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
        value,
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
    res.value.convergence_status && res.variables !== nothing ||
        throw(ArgumentError("Cannot construct SolvedModel: HJB did not converge after $(res.iter) iterations"))
    SolvedModel(m, res.value, res.variables)
end


function SolvedModel(m::Union{StochasticSkibaAbilityModel,StochasticNPAbilityModel}, value::Value, variables::NamedTuple)
    c_interpolation = interpolate(
        (variables.k[:, 1, 1], variables.z[1, :, 1], variables.η[1, 1, :]),
        variables.c,
        Gridded(Linear())
    )
    c_policy_function = extrapolate(c_interpolation, Line())
    prod_func = (k, z, η) -> production_function(m, k, z, η)
    prod_func_prime = (k, z, η) -> production_function_prime(m, k, z, η)
    # need to .dot the policy function or it will give a matrix (Nk x Nz) even
    # if Nk == Nz
    kdot_function = (k, z, η) -> prod_func(k, z, η) + get(variables, :flat_transfer, 0.0) - m.δ*k - c_policy_function.(k, z, η)
    ydot_function = (k, z, η, kdot) -> throw("Not yet implemented - need to add z evolution")
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
        [:k, :z, :η],
        [:c],
        variables,
        value,
        prod_func,
        prod_func_prime,
        c_policy_function,
        kdot_function,
        ydot_function,
        cdot_function,
        m
    )
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

function show(io::IO, r::SolvedModel{T})  where {T <: Union{StochasticSkibaAbilityModel,StochasticNPAbilityModel}}
    median_idx = round(Int, size(r.variables.k, 2) / 2)
    median_3rd_idx = round(Int, size(r.variables.k, 3) / 2)
    print(
        io,
        lineplot(
            r.variables.k[:, median_idx, median_3rd_idx],
            r.variables.c[:, median_idx, median_3rd_idx],
            xlabel = "k(t)",
            ylabel = "c(t)"
        )
    )
end



function plot_model(m::Union{StochasticSkibaAbilityModel,StochasticNPAbilityModel}, value::Value, variables::NamedTuple)
    (; k, z, y, c, η) = variables
    (; v, dVf, dVb, dV0, dist) = value
    fit_kdot = GrowthModels.statespace_k_dot(m)(variables)

    η_reshape = reshape(η[1, 1, :], 1, 1, size(η, 3))
    y = production_function(m, k[:, 1, 1], z[1, :, 1]', η_reshape)

    k_single_vec = k[:, 1, 1]
    med_z_idx = size(k, 2) ÷ 2
    p1 = plot(
        k_single_vec,
        y[:, med_z_idx, :]
    )
    p2 = plot(k_single_vec, v[:, med_z_idx , :], label="V")
    plot!(legend = false)
    xlabel!(p2, "\$k\$")
    ylabel!(p2, "\$v(k)\$")
    p3 = plot(k_single_vec, c[:, med_z_idx, :], label="")
    xlabel!(p3, "\$k\$")
    ylabel!(p3, "\$c(k)\$")

    p4 = plot(k_single_vec, fit_kdot[:, med_z_idx, :], label="")
    hline!(p4, [0], linestyle = :dash, label="kdot = 0")
    xlabel!(p4, "\$k\$")
    ylabel!(p4, "\$s(k)\$")

    subplot = plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))

    return subplot
end



function plot_model(sm::SolvedModel)
    plot_model(sm.m, sm.value, sm.variables)
end
SolvedModel(m::Model, value::Value, ::Nothing) = throw(ArgumentError("Cannot construct SolvedModel from an unconverged HJB solve"))
