# A collateralized credit market for continuous-time one-capital models.
#
# Households borrow against capital, so k <= λa, and choose capital statically given
# wealth. That choice only changes the income the household problem sees: the HJB,
# generator, and stationary code are reused as they are. The scalar rate r lives in
# the outer loop, so fixed abilities remain separate blocks of each inner solve.

"""
    CreditModel(model; λ, r)

The household problem of `model` at a fixed interest rate `r` with collateral limit
`k ≤ λ*a`. A pledgeable share `θ` of capital corresponds to `λ = 1/(1-θ)`, read as a
reduced-form limited-enforcement friction. `λ=1` is no borrowing, and at a low enough
`r` it reproduces `model`. Require `λ ≥ 1` and `r > -δ`.

**The `k` state axis of a `CreditModel` is net worth `a`, not capital.** Capital is the
static choice given by [`capital_choice`](@ref). Wealth follows
`ȧ = profit + r*a - c`, which `capital_drift` reports, and `production` is output at the
chosen capital.

Skiba and Ramsey production are supported, with Ornstein-Uhlenbeck productivity and
fixed ability where present, including a `ResourceModel` that only changes the population. A fixed `r` is a small open economy; use
[`credit_equilibrium`](@ref) to find the rate that clears the market. The Skiba offset
`κ` is ordinary capital here: fully reversible and pledgeable like any other.
"""
struct CreditModel{T,M} <: Model{T}
    model::M
    λ::T
    r::T
    CreditModel{T,M}(::OwnedSnapshot,m,λ,r) where {T,M} = new{T,M}(m,λ,r)
end
function CreditModel(m::Model; λ, r)
    credit_technologies(m)
    isfinite(λ) && primal(λ) >= 1 || throw(ArgumentError("λ must be finite and at least one"))
    isfinite(r) && primal(r)+primal(preferences(m).δ) > 0 ||
        throw(ArgumentError("r must be finite and above -δ, so that capital has a positive user cost"))
    T = promote_type(model_eltype(m),typeof(λ),typeof(r))
    return CreditModel{T,typeof(m)}(OWNED,deepcopy(m),T(λ),T(r))
end
base_model(m::CreditModel) = base_model(m.model)
process(m::CreditModel) = process(m.model)
population(m::CreditModel) = population(m.model)
preferences(m::CreditModel) = preferences(m.model)

# Each technology produces scale*A*max(k-κ,0)^α. The first must have κ=0, so that
# some capital choice is always feasible.
credit_technologies(m::Model) = throw(ArgumentError(
    "CreditModel supports Skiba and Ramsey production with Ornstein-Uhlenbeck productivity, not $(nameof(typeof(m)))"))
skiba_technologies(m) = ((A=m.A_L,κ=zero(m.κ)),(A=m.A_H,κ=m.κ))
ramsey_technologies(m) = ((A=m.A,κ=zero(m.A)),)
# A wrapper can change population, but transfers would need their own treatment here.
function credit_technologies(m::ResourceModel)
    iszero(m.transfer) && m.income_floor == -Inf ||
        throw(ArgumentError("CreditModel does not yet support ResourceModel transfers or income floors"))
    return credit_technologies(m.model)
end
credit_technologies(m::SkibaModel) = skiba_technologies(m)
credit_technologies(m::RamseyCassKoopmansModel) = ramsey_technologies(m)
credit_technologies(m::StochasticSkibaModel{T,<:OrnsteinUhlenbeckProcess}) where T = skiba_technologies(m)
credit_technologies(m::StochasticSkibaAbilityModel{T,<:OrnsteinUhlenbeckProcess}) where T = skiba_technologies(m)
credit_technologies(m::StochasticRamseyCassKoopmansModel{T,<:OrnsteinUhlenbeckProcess}) where T = ramsey_technologies(m)

"""
    capital_choice(model::CreditModel, state)

Profit-maximizing capital at `state`, whose `k` is net worth. Returns `(capital, profit,
technology)`, where `profit = output - (r+δ)*capital` and `technology` indexes the
technology used: 1 is low, 2 is the Skiba high technology. Ties choose the lower index.
"""
function capital_choice(m::CreditModel,s::NamedTuple)
    base = base_model(m)
    scale = (process(m) === nothing ? one(s.k) : s.z)*(population(m) === nothing ? one(s.k) : s.ability)
    cost = m.r+preferences(m).δ
    limit = m.λ*s.k
    technology_choice(tech,index) = begin
        k = min(limit,tech.κ+(base.α*tech.A*scale/cost)^(1/(1-base.α)))
        (capital=k,profit=scale*tech.A*(k-tech.κ)^base.α-cost*k,technology=index)
    end
    techs = credit_technologies(base)
    best = technology_choice(techs[1],1)
    for index in 2:length(techs)
        techs[index].κ < limit || continue
        candidate = technology_choice(techs[index],index)
        candidate.profit > best.profit && (best = candidate)
    end
    return best
end
production(m::CreditModel,s::NamedTuple) = production(m.model,merge(s,(k=capital_choice(m,s).capital,)))
# The solver subtracts δ*a to get drift, leaving profit + r*a before consumption.
resources(m::CreditModel,s::NamedTuple) = capital_choice(m,s).profit+(m.r+preferences(m).δ)*s.k
marginal_product(m::CreditModel,s::NamedTuple) = marginal_product(m.model,merge(s,(k=capital_choice(m,s).capital,)))

# Wealth outruns capital as r nears ρ, so the capital proposal is only a starting point.
const CREDIT_WEALTH_RANGE = 10
function default_grid_spec(m::CreditModel)
    spec = default_numerical_grid_spec(m)
    lo,hi = spec.k.bounds
    return merge(spec,(k=(n=spec.k.n,bounds=(lo/primal(m.λ),CREDIT_WEALTH_RANGE*hi),spacing=:stretched),))
end

"""Outcome of [`credit_equilibrium`](@ref). `scan` summarizes every scanned rate.
`equilibria` holds each market-clearing rate with its `fit` and `distribution`.
`discontinuities` lists brackets where excess demand changes sign without clearing: `reason`
is `:jump` when narrowed to a jump, or the status of a midpoint that could not be computed."""
struct CreditEquilibrium{M,E,D,S}
    model::M
    λ::Float64
    scan::Vector{S}
    equilibria::E
    discontinuities::D
    settings::NamedTuple
    CreditEquilibrium(::OwnedSnapshot,m,λ,scan::Vector{S},e,d,settings) where S =
        new{typeof(m),typeof(e),typeof(d),S}(m,λ,scan,e,d,settings)
end
Base.propertynames(::CreditEquilibrium) = (:model,:λ,:scan,:equilibria,:discontinuities,:settings)
function Base.getproperty(eq::CreditEquilibrium,key::Symbol)
    key == :model && return deepcopy(getfield(eq,:model))
    value = getfield(eq,key)
    return value isa AbstractArray ? _readonly(value) : value
end
function show(io::IO,eq::CreditEquilibrium)
    rates = [round(e.rate;sigdigits=5) for e in getfield(eq,:equilibria)]
    print(io,"CreditEquilibrium(λ=",getfield(eq,:λ),", clearing rates=",rates,
        ", discontinuities=",length(getfield(eq,:discontinuities)),")")
end

"""
    clearing_rate(equilibrium)

The market-clearing rate, if it is unique and wealth stays inside the grid. Otherwise
throws, so that multiple or unreliable equilibria are not silently reduced to a number.
"""
function clearing_rate(eq::CreditEquilibrium)
    found = getfield(eq,:equilibria)
    length(found) == 1 || throw(ArgumentError("Expected one clearing rate, found $(length(found)); inspect equilibria, discontinuities, and scan"))
    only(found).boundary_ok || throw(ArgumentError("Wealth reaches the top of the grid (upper mass $(only(found).upper_mass)); raise the upper bound"))
    return only(found).rate
end

# One outer-loop evaluation. A failure at one rate is a result to report, not a reason
# to discard the rest of the scan, so only the household problem's own errors propagate.
const CREDIT_LATE_WINDOW = 10.0
function credit_market_state(base,grid,λ,r,initial,warm,tolerances,options,long_run)
    m = CreditModel(base;λ=λ,r=r)
    fit = solve(m,grid;initial=warm,options...)
    # A warm start must never decide the answer, so it can only save time.
    if !fit.converged && warm !== nothing
        fit = solve(m,grid;options...)
    end
    failed(status,detail) = (summary=(rate=r,status=status,detail=detail,excess=NaN,relative_excess=NaN,
        capital=NaN,wealth=NaN,borrowing=NaN,output=NaN,upper_mass=NaN,boundary_ok=false,late_change=NaN),fit=fit,distribution=nothing)
    fit.converged || return failed(:not_converged,string(fit.diagnostics.stop_reason))
    late_change = NaN
    d = try
        if long_run.horizon === nothing
            stationary(fit,initial)
        else
            # Distribution after `horizon` years at this fixed rate, plus a check of how
            # much it was still moving over the final decade.
            path = trajectory(Evolution(fit;dt=long_run.dt),initial,[long_run.horizon-CREDIT_LATE_WINDOW,long_run.horizon])
            late_change = sum(abs,view(path.probabilities,:,2).-view(path.probabilities,:,1))/2
            final(path)
        end
    catch err
        err isa ArgumentError || rethrow()
        return failed(:no_stationary,sprint(showerror,err))
    end
    capital = expectation(d,s -> capital_choice(m,s).capital)
    wealth = expectation(d,s -> s.k)
    borrowing = expectation(d,s -> max(capital_choice(m,s).capital-s.k,0.0))
    output = expectation(d,s -> production(m,s))
    upper = boundary_mass(d).upper
    summary = (rate=r,status=:solved,detail="",excess=capital-wealth,relative_excess=(capital-wealth)/wealth,
        capital=capital,wealth=wealth,borrowing=borrowing,output=output,upper_mass=upper,
        boundary_ok=upper <= tolerances.boundary,late_change=late_change)
    return (summary=summary,fit=fit,distribution=d)
end

function refine_crossing(evaluate,lower,upper,tolerances)
    best = abs(lower.relative_excess) <= abs(upper.relative_excess) ? lower : upper
    warm = nothing
    while abs(best.relative_excess) > tolerances.clearing && upper.rate-lower.rate > tolerances.rate
        state = evaluate((lower.rate+upper.rate)/2,warm)
        point = state.summary
        # A failed midpoint leaves the bracket unresolved, and it is reported as such.
        point.status == :solved || return (best=best,lower=lower,upper=upper,failure=point.status)
        warm = state.fit
        abs(point.relative_excess) < abs(best.relative_excess) && (best = point)
        if signbit(point.excess) == signbit(lower.excess)
            lower = point
        else
            upper = point
        end
    end
    return (best=best,lower=lower,upper=upper,failure=nothing)
end

"""
    credit_equilibrium(model, grid; λ, rates=nothing, initial=nothing, horizon=nothing, dt=1.0,
        rate_tolerance=1e-5, clearing_tolerance=1e-3, boundary_tolerance=1e-4,
        solve_options=NamedTuple())

Find stationary rates clearing the credit market, `∫(k-a)dμ = 0`, with one rate in a
closed economy. For each rate in `rates` it solves the household problem and its
stationary distribution, then refines every sign change of excess demand by bisection.
By default `rates` scans `(-δ, ρ)`, the range where capital has a positive user cost and
wealth stays bounded. Scanning is what reveals multiple equilibria, but only if `rates`
is fine enough to separate them.

By default the market clears on the stationary distribution. With `horizon`, it clears
on the distribution `horizon` years after `initial` at that fixed rate, as in Buera and
Shin (2013). The rate is held fixed along the way, so this approximates a long-run
equilibrium; it is not a transition path. `horizon` requires `initial`, since the result
depends on where households start, and `late_change` reports the total-variation change
over the final ten years. Use a horizon when some states are metastable: escape is then
possible but so slow that the stationary distribution cannot be computed, and the scan
reports `:no_stationary` at those rates.

Requires productivity risk, because below `ρ` only precautionary saving creates lenders,
and `λ > 1`, because with no borrowers the rate is not determined. Population shares now
affect policies through the rate. `initial` selects masses among closed classes when the
stationary distribution is not unique.

A sign change clears when `|K-A|/A ≤ clearing_tolerance`. Otherwise it is reported in
`discontinuities`, since excess demand can jump: unconstrained households at one
productivity state all switch technology at the same rate. Such jumps shrink as `z` is
refined, and clearing rates converge slowly in `z`. `boundary_ok` is false when mass above
`boundary_tolerance` sits in the top wealth cell, which makes the result unreliable even
though the solve converged. Check clearing rates on a refined grid. The rate is not
differentiated: this routine rejects automatic differentiation.
"""
function credit_equilibrium(base::Model,grid::StateGrid; λ,rates=nothing,initial=nothing,horizon=nothing,dt=1.0,
        rate_tolerance=1e-5,clearing_tolerance=1e-3,boundary_tolerance=1e-4,solve_options=NamedTuple())
    credit_technologies(base)
    process(base) === nothing && throw(ArgumentError(
        "credit_equilibrium requires productivity risk: without it nobody lends below r=ρ, and at r=ρ the wealth distribution is indeterminate"))
    λ isa ForwardDiff.Dual && throw(ArgumentError("credit_equilibrium does not support automatic differentiation"))
    isfinite(λ) && λ > 1 || throw(ArgumentError("credit_equilibrium requires a finite λ above one: with λ=1 nobody can borrow and the rate is not determined"))
    all(x -> isfinite(x) && x > 0,(rate_tolerance,clearing_tolerance,boundary_tolerance)) ||
        throw(ArgumentError("Tolerances must be finite and positive"))
    haskey(solve_options,:initial) && throw(ArgumentError("The outer loop manages warm starts; do not pass initial in solve_options"))
    if horizon !== nothing
        initial === nothing && throw(ArgumentError("A finite horizon needs an explicit initial distribution: the result depends on where households start"))
        horizon isa Real && isfinite(horizon) && horizon > CREDIT_LATE_WINDOW ||
            throw(ArgumentError("horizon must be a finite number of years above $CREDIT_LATE_WINDOW"))
        dt isa Real && isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    end
    long_run = (horizon=horizon === nothing ? nothing : Float64(horizon),dt=Float64(dt))
    if initial !== nothing
        compatible_grid(grid,initial.grid) || throw(ArgumentError("Initial distribution grid differs from grid"))
    end
    p = preferences(base)
    ρ,δ = primal(p.ρ),primal(p.δ)
    width = ρ+δ
    scan_rates = rates === nothing ? collect(range(-δ+.05width,ρ-.02width;length=13)) : sort!(float.(collect(rates)))
    length(scan_rates) >= 2 && all(isfinite,scan_rates) && allunique(scan_rates) ||
        throw(ArgumentError("rates must contain at least two distinct finite values"))
    all(r -> -δ < r < ρ,scan_rates) || throw(ArgumentError("rates must lie strictly between -δ and ρ"))

    tolerances = (rate=rate_tolerance,clearing=clearing_tolerance,boundary=boundary_tolerance)
    evaluate(r,warm) = credit_market_state(base,grid,λ,r,initial,warm,tolerances,solve_options,long_run)
    scan = NamedTuple[]
    warm = nothing
    for r in scan_rates
        state = evaluate(r,warm)
        state.summary.status == :solved && (warm = state.fit)
        push!(scan,state.summary)
    end
    scan = [point for point in scan]

    equilibria,discontinuities = [],[]
    for (lower,upper) in zip(scan[1:end-1],scan[2:end])
        lower.status == upper.status == :solved && signbit(lower.excess) != signbit(upper.excess) || continue
        bracket = refine_crossing(evaluate,lower,upper,tolerances)
        if abs(bracket.best.relative_excess) <= clearing_tolerance
            # Solving once more without a warm start keeps the reported fit independent
            # of the search path that found this rate.
            state = evaluate(bracket.best.rate,nothing)
            state.summary.status == :solved && abs(state.summary.relative_excess) <= clearing_tolerance &&
                (push!(equilibria,merge(state.summary,(fit=state.fit,distribution=state.distribution))); continue)
        end
        # A jump narrowed to the rate tolerance, or a bracket a failed midpoint left open.
        push!(discontinuities,(lower_rate=bracket.lower.rate,upper_rate=bracket.upper.rate,
            lower_excess=bracket.lower.excess,upper_excess=bracket.upper.excess,
            reason=bracket.failure === nothing ? :jump : bracket.failure))
    end
    settings = (rate_tolerance=rate_tolerance,clearing_tolerance=clearing_tolerance,
        boundary_tolerance=boundary_tolerance,rates=copy(scan_rates),
        long_run=horizon === nothing ? :stationary : :horizon,horizon=long_run.horizon,dt=long_run.dt)
    return CreditEquilibrium(OWNED,deepcopy(base),Float64(λ),scan,[e for e in equilibria],[d for d in discontinuities],settings)
end
