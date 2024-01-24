
struct HyperParams
    N::Int64
    dk::Float64
    kmax::Float64
    kmin::Float64
end


# Create a HyperParams object 
# just take kmin and kmax as inputs to create grid
function HyperParams(N = 1000, kmax = 10, kmin = 0.001)
    dk = (kmax-kmin)/(N-1)
    HyperParams(N, dk, kmax, kmin)
end


struct Value
    v::Array{Float64,1}
    dVf::Array{Float64,1}
    dVb::Array{Float64,1}
    dV0::Array{Float64,1}
    dist::Array{Float64,1}
    convergence_status::Bool
    iter::Int64
end

function Value(hyperparams::HyperParams)
    v = zeros(hyperparams.N)
    dVf = zeros(hyperparams.N)
    dVb = zeros(hyperparams.N)
    dV0 = zeros(hyperparams.N)
    dist = fill(Inf, hyperparams.N)
    convergence_status = false
    iter = 0
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end

function Value(; v, dVf, dVb, dV0, dist, convergence_status = false, iter = 0)
    Value(v, dVf, dVb, dV0, dist, convergence_status, iter)
end



#### Diagnostic Plotting ####

function plot_diagnostics(m::Model, value::Value, variables::NamedTuple, hyperparams::HyperParams)
    Verr = V_err(m)(value, variables);
    convergence_status, curr_iter = value.convergence_status, value.iter
    subplot = plot(
        layout = (1, 2),
        size = (800, 600)
    ) 
    plot!(
        subplot[1],
        log.(value.dist[1:curr_iter]), 
        label = "\$||V^{n+1} - V^n||\$", 
        xlabel = "Iteration", 
        ylabel = "log(Distance)"
    )
    plot!(
        subplot[2],
        variables.k, 
        Verr, 
        linewidth=2, 
        label="", 
        xlabel="k", 
        ylabel="Error in HJB Equation",
        xlims=(hyperparams.kmin, hyperparams.kmax)
    )
    title!(
        subplot, 
        "Convergence Diagnostics - Status: $(convergence_status)"
        ) 

    return subplot
end

function plot_model(m::Model, value::Value, variables::NamedTuple)
    (; k, y, c) = variables
    (; v, dVf, dVb, dV0, dist) = value
    kstar = k_star(m)
    fit_kdot = k_dot(m)(variables)

    # subplot = plot(layout = (2, 2), size = (800, 600))
    p1 =  plot_production_function(m, k)
    scatter!(p1, [kstar], [production_function(m)(kstar)], label="kstar", markersize=4)

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