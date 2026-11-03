"""Fixed economic types and their default population probabilities."""
struct Population{V,P}
    values::V
    shares::P
    function Population(values, shares)
        v, p = collect(values), collect(shares)
        length(v) == length(p) > 0 || throw(DimensionMismatch("Population values and shares must have the same nonzero length"))
        all(isfinite, v) && all(diff(v) .> 0) || throw(ArgumentError("Population values must be finite and strictly increasing"))
        all(x -> isfinite(x) && x >= 0, p) && abs(sum(p)-1) <= 1e-10 || throw(ArgumentError("Population shares must be nonnegative and sum to one"))
        new{typeof(v),typeof(p)}(copy(v), copy(p))
    end
end

base_model(m::Model) = m
is_deterministic(m::Model) = m isa DeterministicModel
population(::Model) = nothing
population(m::Union{StochasticSkibaAbilityModel,StochasticNPAbilityModel}) =
    Population(m.η_mean .* [0.8, 0.9, 1.0, 1.1, 1.2], fill(0.2, 5))
preferences(m::Model) = (γ=m.γ, ρ=m.ρ, δ=m.δ)
state_schema(m::Model) = is_deterministic(m) ? (:k,) : population(m) === nothing ? (:k, :z) : (:k, :z, :ability)
production(m::DeterministicModel, state::NamedTuple) = production_function(m, state.k)
production(m::StochasticModel, state::NamedTuple) = population(m) === nothing ?
    production_function(m, state.k, state.z) : production_function(m, state.k, state.z, state.ability)
resources(m::Model, state::NamedTuple) = production(m, state)
marginal_product(m::Model, state::NamedTuple) = ForwardDiff.derivative(k -> production(m, merge(state, (k=k,))), state.k)

"""
    ResourceModel(model; transfer=0, income_floor=-Inf, population=population(model))

An economic specification with disposable income `max(production + transfer,
income_floor)`, before depreciation and consumption. The production model and
population are owned snapshots. Custom model types can instead implement
`resources(model, state)` directly.
"""
struct ResourceModel{T,M,P} <: Model{T}
    model::M
    transfer::T
    income_floor::T
    population::P
end
function ResourceModel(m::Model; transfer=0.0, income_floor=-Inf, population=population(m))
    T = promote_type(model_eltype(m), typeof(transfer), typeof(income_floor))
    isfinite(transfer) && !isnan(income_floor) && income_floor < Inf || throw(ArgumentError("Transfer must be finite and income floor finite or -Inf"))
    (population === nothing) == (GrowthModels.population(m) === nothing) || throw(ArgumentError("ResourceModel cannot change the model's state schema"))
    return ResourceModel{T,typeof(m),typeof(population)}(deepcopy(m), T(transfer), T(income_floor), deepcopy(population))
end
function Base.getproperty(m::ResourceModel, name::Symbol)
    name in fieldnames(typeof(m)) && return getfield(m, name)
    return getproperty(getfield(m, :model), name)
end
base_model(m::ResourceModel) = base_model(m.model)
is_deterministic(m::ResourceModel) = is_deterministic(m.model)
population(m::ResourceModel) = m.population
preferences(m::ResourceModel) = preferences(m.model)
production(m::ResourceModel, state::NamedTuple) = production(m.model, state)
resources(m::ResourceModel, state::NamedTuple) = max(production(m, state) + m.transfer, m.income_floor)

"""A one-capital model with concrete callable production `f(k,z,ability)`."""
struct FunctionalModel{T,F,S,P} <: Model{T}
    γ::T
    ρ::T
    δ::T
    f::F
    stochasticprocess::S
    population::P
end
function FunctionalModel(f, process::StochasticProcess, types::Population; γ=2.0, ρ=0.05, δ=0.05)
    γ, ρ, δ = promote(γ, ρ, δ)
    return FunctionalModel(γ, ρ, δ, deepcopy(f), deepcopy(process), deepcopy(types))
end
population(m::FunctionalModel) = m.population
production(m::FunctionalModel, state::NamedTuple) = m.f(state.k, state.z, state.ability)
is_deterministic(::FunctionalModel) = false

# Defaults are proposals for numerical resolution, never population definitions.
default_grid_spec(m::Model) = default_numerical_grid_spec(m)
default_grid_spec(m::ResourceModel) = default_grid_spec(m.model)
default_grid_spec(m::ResourceModel{T,M}) where {T,M<:StochasticSkibaAbilityModel} = default_numerical_grid_spec(m)
function default_numerical_grid_spec(m::Model)
    b = base_model(m)
    if b isa FunctionalModel || b isa StochasticNPAbilityModel
        capital = (n=1000, bounds=(0.001, 10.0), spacing=:linear)
    elseif is_deterministic(m)
        capital = (n=1000, bounds=(0.001*k_steady_state_hi(b), 1.3*k_steady_state_hi(b)), spacing=:linear)
    else
        p = m.stochasticprocess
        upper_z = p isa OrnsteinUhlenbeckProcess ? last(productivity_bounds(p)) : maximum(p.z)
        upper_type = population(m) === nothing ? one(upper_z) : maximum(population(m).values)
        capital = (n=1000, bounds=(0.001*k_steady_state_hi(b), 1.3*capital_reference(b, upper_z, upper_type)), spacing=:linear)
    end
    return is_deterministic(m) ? (k=capital,) : (k=capital, z=(n=40, width=4))
end
