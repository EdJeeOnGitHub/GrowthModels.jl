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
        risk_aversion=2.0, A=1.0, α=0.3, productivity=nothing, population=nothing)

Native growth with production `A*k^α` and budget
`c + k′ = production + (1-depreciation)*k`. Investment is reversible and
capital is nonnegative. All rates are per period. Optional `productivity`
(`MarkovChain` or `LogAR1`) and `population` (`Population`) multiply production
by current productivity and fixed ability. Productivity is observed before saving.
Risk aversion one uses log utility. The discrete solver uses Float64 and does not support AD.
"""
struct DiscreteRamseyCassKoopmansModel{S,P} <: DiscreteTimeModel
    discount::Float64
    depreciation::Float64
    risk_aversion::Float64
    A::Float64
    α::Float64
    productivity::S
    population::P
    function DiscreteRamseyCassKoopmansModel(; discount=0.96, depreciation=0.08,
            risk_aversion=2.0, A=1.0, α=0.3, productivity=nothing, population=nothing)
        β, δ, γ, α, A = discrete_parameters(discount, depreciation, risk_aversion, α, A)
        A > 0 || throw(ArgumentError("A must be positive"))
        validate_discrete_heterogeneity(productivity,population)
        new{typeof(productivity),typeof(population)}(β, δ, γ, A, α, deepcopy(productivity), deepcopy(population))
    end
end

"""
    DiscreteSkibaModel(; discount=0.96, depreciation=0.08, risk_aversion=2.0,
        A_H=0.6, A_L=0.4, α=0.3, κ=2.0, productivity=nothing, population=nothing)

Native growth with the best production technology each period:
`max(A_L*k^α, A_H*max(k-κ,0)^α)`. The capital offset κ is not an adoption
payment. Optional productivity and fixed ability multiply both technologies equally.
Budget, reversibility, utility, and precision follow
[`DiscreteRamseyCassKoopmansModel`](@ref). Require `A_H > A_L > 0` and `κ >= 0`.
"""
struct DiscreteSkibaModel{S,P} <: DiscreteTimeModel
    discount::Float64
    depreciation::Float64
    risk_aversion::Float64
    A_H::Float64
    A_L::Float64
    α::Float64
    κ::Float64
    productivity::S
    population::P
    function DiscreteSkibaModel(; discount=0.96, depreciation=0.08,
            risk_aversion=2.0, A_H=0.6, A_L=0.4, α=0.3, κ=2.0, productivity=nothing, population=nothing)
        β, δ, γ, α, ah, al, κ = discrete_parameters(discount, depreciation,
            risk_aversion, α, A_H, A_L, κ)
        ah > al > 0 || throw(ArgumentError("Require A_H > A_L > 0"))
        κ >= 0 || throw(ArgumentError("κ must be nonnegative"))
        validate_discrete_heterogeneity(productivity,population)
        new{typeof(productivity),typeof(population)}(β, δ, γ, ah, al, α, κ, deepcopy(productivity), deepcopy(population))
    end
end

function validate_discrete_heterogeneity(p,types)
    p === nothing || p isa DiscreteProductivity ||
        throw(ArgumentError("Use MarkovChain or LogAR1 for native discrete-time productivity"))
    types === nothing || types isa Population || throw(ArgumentError("Ability types require a Population"))
    if types !== nothing
        all(x -> !(x isa ForwardDiff.Dual) && isfinite(x) && x > 0 && isfinite(Float64(x)) && Float64(x)>0,types.values) &&
            all(x -> !(x isa ForwardDiff.Dual),types.shares) ||
            throw(ArgumentError("Discrete ability must be positive and non-AD"))
    end
end
process(m::DiscreteTimeModel) = m.productivity
population(m::DiscreteTimeModel) = m.population
function state_schema(m::DiscreteTimeModel)
    p,a=process(m),population(m)
    return p === nothing ? (a === nothing ? (:k,) : (:k,:ability)) :
        (a === nothing ? (:k,:z) : (:k,:z,:ability))
end
function state_schema(m::ResourceModel)
    base_model(m) isa DiscreteTimeModel && return state_schema(base_model(m))
    return invoke(state_schema,Tuple{Model},m)
end
preferences(m::DiscreteTimeModel) =
    (discount=m.discount, depreciation=m.depreciation, risk_aversion=m.risk_aversion)
function discrete_scale(m,s)
    process(m) !== nothing && !haskey(s,:z) && throw(ArgumentError("Productivity state requires z"))
    population(m) !== nothing && !haskey(s,:ability) && throw(ArgumentError("Ability state requires ability"))
    return (process(m) === nothing ? 1.0 : s.z)*(population(m) === nothing ? 1.0 : s.ability)
end
production(m::DiscreteRamseyCassKoopmansModel, s::NamedTuple) = discrete_scale(m,s)*m.A*s.k^m.α
production(m::DiscreteSkibaModel, s::NamedTuple) =
    discrete_scale(m,s)*skiba_production_function(s.k,m.α,m.A_H,m.A_L,m.κ)
technology_crossing(m::DiscreteSkibaModel) = k_star_Skiba(m.α,m.A_H,m.A_L,m.κ)
discrete_budget(m,s::NamedTuple) = resources(m,s)+(1-preferences(m).depreciation)*s.k
discrete_budget(m,k::Real) = discrete_budget(m,(k=k,))

function discrete_capital_spec(m,z,ability)
    b=base_model(m)
    cost=1/b.discount-1+b.depreciation
    scale=maximum(z)*maximum(ability)
    reference = if b isa DiscreteRamseyCassKoopmansModel
        (b.α*b.A*scale/cost)^(1/(1-b.α))
    else
        low=(b.α*b.A_L*scale/cost)^(1/(1-b.α))
        high=(b.α*b.A_H*scale/cost)^(1/(1-b.α))+b.κ
        crossing=technology_crossing(b)
        max(low <= crossing ? low : 0.0,high >= crossing ? high : 0.0,crossing)
    end
    return (n=500,bounds=(max(1e-8,0.001*reference),1.3*reference),spacing=:linear)
end
function default_grid_spec(m::DiscreteTimeModel)
    p=process(m)
    z=p === nothing ? [1.0] : process_axis(p,default_grid_spec(p)).coordinates
    ability=population(m) === nothing ? [1.0] : population(m).values
    k=discrete_capital_spec(m,z,ability)
    return p === nothing ? (k=k,) : (k=k,z=default_grid_spec(p))
end

function discrete_kinks(m::DiscreteTimeModel,s=(k=0.0,))
    m isa DiscreteSkibaModel ? [technology_crossing(m)] : Float64[]
end
function discrete_kinks(m::ResourceModel,s=(k=0.0,))
    knots = discrete_kinks(m.model,s)
    # Both supplied technologies are continuous and strictly increasing. Locate
    # a floor crossing independently of the numerical domain.
    target = m.income_floor - m.transfer
    if isfinite(target) && target > resources(m.model, merge(s,(k=0.0,)))
        lo, hi = 0.0, 1.0
        while resources(m.model, merge(s,(k=hi,))) < target && isfinite(2hi)
            hi *= 2
        end
        if resources(m.model, merge(s,(k=hi,))) >= target
            for _ in 1:100
                mid = lo + (hi-lo)/2
                resources(m.model, merge(s,(k=mid,))) < target ? (lo=mid) : (hi=mid)
            end
            push!(knots, lo + (hi-lo)/2)
        end
    end
    return knots
end

"""Discrete grids allow zero capital. Generated grids include known resource
kinks; explicitly supplied coordinates remain exactly as supplied."""
function discrete_grid(m; k=NamedTuple(), z=nothing, kwargs...)
    isempty(kwargs) || throw(ArgumentError("Only k/z grids may be supplied; ability support belongs to the model"))
    p,types=process(m),population(m)
    p === nothing && z !== nothing && throw(ArgumentError("Model has no productivity axis"))
    zspec=p === nothing ? nothing : (z === nothing ? default_grid_spec(p) : z)
    zs=p === nothing ? [1.0] : process_axis(p,zspec).coordinates
    abilities=types === nothing ? [1.0] : types.values
    if k isa NamedTuple
        validate_options(k, (:n,:bounds,:spacing,:power), "capital axis")
        spec = haskey(k,:n) && haskey(k,:bounds) ? k : merge(discrete_capital_spec(m,zs,abilities), k)
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
        for productivity in zs, ability in abilities
            state=(k=0.0,z=productivity,ability=ability)
            append!(axis,filter(x -> lo < x < hi,discrete_kinks(m,state)))
        end
        sort!(unique!(axis))
    else
        spec = deepcopy(k)
        any(x -> x isa ForwardDiff.Dual, k) && throw(ArgumentError("Discrete grids do not support AD"))
        axis = Float64.(collect(k))
    end
    length(axis) >= 2 && all(isfinite,axis) && first(axis) >= 0 && all(diff(axis) .> 0) ||
        throw(ArgumentError("Capital requires at least two finite, nonnegative, increasing points"))
    axes,kinds,specs=(k=axis,),(k=:interpolable,),(k=deepcopy(spec),)
    if p !== nothing
        axes=merge(axes,(z=Float64.(zs),))
        kinds=merge(kinds,(z=:categorical,))
        specs=merge(specs,(z=deepcopy(zspec),))
    end
    if types !== nothing
        axes=merge(axes,(ability=Float64.(abilities),))
        kinds=merge(kinds,(ability=:categorical,))
    end
    dims=map(length,values(axes))
    weights=reshape(create_dx_stacked(axis,prod(dims) ÷ length(axis)),dims)
    return StateGrid(OWNED,axes,kinds,specs,weights)
end

StateGrid(m::DiscreteTimeModel; kwargs...) = discrete_grid(m; kwargs...)
function StateGrid(m::ResourceModel; kwargs...)
    base_model(m) isa DiscreteTimeModel && return discrete_grid(m; kwargs...)
    return invoke(StateGrid, Tuple{Model}, m; kwargs...)
end
