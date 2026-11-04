"""Native discrete-time economics; parameters are per period, with no AD contract."""
abstract type DiscreteTimeModel <: Model{Float64} end

function discrete_parameters(xs...)
    all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x), xs) ||
        throw(ArgumentError("Discrete-time parameters must be finite real numbers; AD is unsupported"))
    ys = Float64.(xs)
    all(isfinite, ys) || throw(ArgumentError("Parameters must be representable as Float64"))
    β, δ, γ, α = ys[1:4]
    0 < β < 1 || throw(ArgumentError("discount must lie strictly between zero and one"))
    0 <= δ <= 1 || throw(ArgumentError("depreciation must lie between zero and one"))
    γ > 0 || throw(ArgumentError("risk_aversion must be positive"))
    0 < α < 1 || throw(ArgumentError("α must lie strictly between zero and one"))
    return ys
end

"""
    DiscreteRamseyCassKoopmansModel(; discount=0.96, depreciation=0.08,
        risk_aversion=2.0, A=1.0, α=0.3)

Native deterministic growth with production `A*k^α` and budget
`c + k′ = production + (1-depreciation)*k`. Investment is reversible and
capital is nonnegative. All rates are per period. Risk aversion one uses log
utility. The discrete solver uses Float64 and does not support AD.
"""
struct DiscreteRamseyCassKoopmansModel <: DiscreteTimeModel
    discount::Float64
    depreciation::Float64
    risk_aversion::Float64
    A::Float64
    α::Float64
    function DiscreteRamseyCassKoopmansModel(; discount=0.96, depreciation=0.08,
            risk_aversion=2.0, A=1.0, α=0.3)
        β, δ, γ, α, A = discrete_parameters(discount, depreciation, risk_aversion, α, A)
        A > 0 || throw(ArgumentError("A must be positive"))
        new(β, δ, γ, A, α)
    end
end

"""
    DiscreteSkibaModel(; discount=0.96, depreciation=0.08, risk_aversion=2.0,
        A_H=0.6, A_L=0.4, α=0.3, κ=2.0)

Native deterministic growth with the best production technology each period:
`max(A_L*k^α, A_H*max(k-κ,0)^α)`. The capital offset κ is not an adoption
payment. Budget, reversibility, utility, and precision follow
[`DiscreteRamseyCassKoopmansModel`](@ref). Require `A_H > A_L > 0` and `κ >= 0`.
"""
struct DiscreteSkibaModel <: DiscreteTimeModel
    discount::Float64
    depreciation::Float64
    risk_aversion::Float64
    A_H::Float64
    A_L::Float64
    α::Float64
    κ::Float64
    function DiscreteSkibaModel(; discount=0.96, depreciation=0.08,
            risk_aversion=2.0, A_H=0.6, A_L=0.4, α=0.3, κ=2.0)
        β, δ, γ, α, ah, al, κ = discrete_parameters(discount, depreciation,
            risk_aversion, α, A_H, A_L, κ)
        ah > al > 0 || throw(ArgumentError("Require A_H > A_L > 0"))
        κ >= 0 || throw(ArgumentError("κ must be nonnegative"))
        new(β, δ, γ, ah, al, α, κ)
    end
end

preferences(m::DiscreteTimeModel) =
    (discount=m.discount, depreciation=m.depreciation, risk_aversion=m.risk_aversion)
production(m::DiscreteRamseyCassKoopmansModel, s::NamedTuple) = m.A * s.k^m.α
production(m::DiscreteSkibaModel, s::NamedTuple) =
    skiba_production_function(s.k, m.α, m.A_H, m.A_L, m.κ)
technology_crossing(m::DiscreteSkibaModel) = k_star_Skiba(m.α, m.A_H, m.A_L, m.κ)
discrete_budget(m, k) = resources(m, (k=k,)) + (1-preferences(m).depreciation)*k

function default_grid_spec(m::DiscreteTimeModel)
    cost = 1/m.discount - 1 + m.depreciation
    reference = if m isa DiscreteRamseyCassKoopmansModel
        (m.α*m.A/cost)^(1/(1-m.α))
    else
        low = (m.α*m.A_L/cost)^(1/(1-m.α))
        high = (m.α*m.A_H/cost)^(1/(1-m.α)) + m.κ
        crossing = technology_crossing(m)
        # Only branch-valid steady states are references. Include the crossing
        # so a bound proposal cannot silently exclude the high technology.
        max(low <= crossing ? low : 0.0, high >= crossing ? high : 0.0, crossing)
    end
    return (k=(n=500, bounds=(max(1e-8, 0.001*reference), 1.3*reference), spacing=:linear),)
end

function discrete_kinks(m::DiscreteTimeModel)
    m isa DiscreteSkibaModel ? [technology_crossing(m)] : Float64[]
end
function discrete_kinks(m::ResourceModel)
    knots = discrete_kinks(m.model)
    # Both supplied technologies are continuous and strictly increasing. Locate
    # a floor crossing independently of the numerical domain.
    target = m.income_floor - m.transfer
    if isfinite(target) && target > resources(m.model, (k=0.0,))
        lo, hi = 0.0, 1.0
        while resources(m.model, (k=hi,)) < target && isfinite(2hi)
            hi *= 2
        end
        if resources(m.model, (k=hi,)) >= target
            for _ in 1:100
                mid = lo + (hi-lo)/2
                resources(m.model, (k=mid,)) < target ? (lo=mid) : (hi=mid)
            end
            push!(knots, lo + (hi-lo)/2)
        end
    end
    return knots
end

"""Discrete grids allow zero capital. Generated grids include known resource
kinks; explicitly supplied coordinates remain exactly as supplied."""
function discrete_grid(m; k=NamedTuple(), kwargs...)
    isempty(kwargs) || throw(ArgumentError("Discrete deterministic models support only the k axis"))
    if k isa NamedTuple
        validate_options(k, (:n,:bounds,:spacing,:power), "capital axis")
        spec = haskey(k,:n) && haskey(k,:bounds) ? k : merge(default_grid_spec(m).k, k)
        n = spec.n
        n isa Integer && n >= 2 || throw(ArgumentError("Capital grid needs at least two points"))
        lo, hi = spec.bounds
        all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x), (lo,hi)) &&
            0 <= lo < hi || throw(ArgumentError("Capital bounds must be finite, nonnegative, and increasing"))
        spacing = get(spec,:spacing,:linear)
        spacing in (:linear,:stretched) || throw(ArgumentError("Unknown capital spacing"))
        power = spacing == :linear ? 1.0 : get(spec,:power,2.0)
        isfinite(power) && power > 0 || throw(ArgumentError("Grid power must be positive"))
        axis = Float64[lo + (hi-lo)*t^power for t in range(0,1;length=n)]
        append!(axis, filter(x -> lo < x < hi, discrete_kinks(m)))
        sort!(unique!(axis))
    else
        spec = deepcopy(k)
        any(x -> x isa ForwardDiff.Dual, k) && throw(ArgumentError("Discrete grids do not support AD"))
        axis = Float64.(collect(k))
    end
    length(axis) >= 2 && all(isfinite,axis) && first(axis) >= 0 && all(diff(axis) .> 0) ||
        throw(ArgumentError("Capital requires at least two finite, nonnegative, increasing points"))
    return StateGrid(OWNED, (k=axis,), (k=:interpolable,), (k=deepcopy(spec),),
        create_dx_stacked(axis,1))
end

StateGrid(m::DiscreteTimeModel; kwargs...) = discrete_grid(m; kwargs...)
function StateGrid(m::ResourceModel; kwargs...)
    base_model(m) isa DiscreteTimeModel && return discrete_grid(m; kwargs...)
    return invoke(StateGrid, Tuple{Model}, m; kwargs...)
end
