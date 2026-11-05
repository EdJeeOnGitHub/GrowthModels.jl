# Discrete-time credit market with committed capital (DISCRETE_CREDIT_DESIGN.md).
#
# Capital is bought one period and exposed to the next period's shock, financed by own
# funds and debt. Lenders can seize only the asset, as in Kiyotaki and Moore (1997):
# (1+r)b′ ≤ θ(1−δ)k′. Because resale value is certain, repayment is always feasible and
# the model needs no default. The state is cash on hand after production and repayment.
#
# Each period solves two stages. Allocation: for own funds s, choose capital k′ ≤ λs to
# maximize expected continuation value; this is a risky choice, so it is not static.
# Saving: choose s given that continuation, which is the existing one-dimensional
# problem solved by `bellman_choice`.

"""
    DiscreteCreditModel(model; θ, r)

Borrowing and lending at a fixed per-period rate `r`, with capital committed before the
next productivity draw. `model` is a `DiscreteSkibaModel` or
`DiscreteRamseyCassKoopmansModel`, including productivity and ability.

**The `k` state axis of this model is cash on hand `x`, not capital.** Each period:

    x  = y(k, z, η) + (1−δ)k − (1+r)b          (state: after production and repayment)
    c  = x − s,  s ∈ [0, x]                     (own funds)
    k′ ∈ [0, λs],  b′ = k′ − s                   (capital and debt, before z′ is known)
    x′ = y(k′, z′, η) − (r+δ)k′ + (1+r)s

The borrowing constraint is `(1+r)b′ ≤ θ(1−δ)k′`: repayment cannot exceed the share `θ`
of the asset's resale value a lender can recover. Equivalently `k′ ≤ λs` with
`λ = (1+r)/(1+r−θ(1−δ))`. Repayment is always feasible, so there is no default and cash
on hand is never negative. Households can lend at `r` (`k′ < s`) but cannot borrow
without collateral. The Skiba offset `κ` is pledgeable like any other capital.

With `θ = 0` and a rate low enough that investing beats lending, `k′ = s` and the model
reproduces `model`. Require `0 ≤ θ ≤ 1` and `r > −δ`.
"""
struct DiscreteCreditModel{M} <: DiscreteTimeModel
    model::M
    θ::Float64
    r::Float64
    DiscreteCreditModel{M}(::OwnedSnapshot,m,θ,r) where M = new{M}(m,θ,r)
end
function DiscreteCreditModel(m::Union{DiscreteSkibaModel,DiscreteRamseyCassKoopmansModel}; θ, r)
    all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x),(θ,r)) ||
        throw(ArgumentError("θ and r must be finite real numbers; AD is unsupported"))
    0 <= θ <= 1 || throw(ArgumentError("θ must lie between zero and one"))
    r > -m.depreciation || throw(ArgumentError("r must exceed -depreciation, so capital has a positive user cost"))
    return DiscreteCreditModel{typeof(m)}(OWNED,deepcopy(m),Float64(θ),Float64(r))
end
base_model(m::DiscreteCreditModel) = base_model(m.model)
process(m::DiscreteCreditModel) = process(m.model)
population(m::DiscreteCreditModel) = population(m.model)
preferences(m::DiscreteCreditModel) = preferences(m.model)
production(m::DiscreteCreditModel,s::NamedTuple) = production(m.model,s)
"""Maximum ratio of capital to own funds implied by the collateral constraint."""
leverage(m::DiscreteCreditModel) = (1+m.r)/(1+m.r-m.θ*(1-preferences(m).δ))
# Cash-on-hand grids have no technology kinks; base-model steady states are in capital.
discrete_kinks(::DiscreteCreditModel,s=(k=0.0,)) = Float64[]
uncovered_steady_states(::DiscreteCreditModel,g) = Float64[]
capital_steady_states(m::DiscreteCreditModel,scale) = capital_steady_states(m.model,scale)

# Precomputed objects for one solve. Values are indexed (cash, productivity, ability).
struct CreditWorkspace
    x::Vector{Float64}            # cash-on-hand grid
    s::Vector{Float64}            # own-funds grid: zero, then the cash grid
    capital::Vector{Float64}      # capital grid for the allocation search
    output::Array{Float64,3}      # y(capital, z′, ability)
    P::Matrix{Float64}            # productivity transition
    zs::Vector{Float64}
    abilities::Vector{Float64}
    λ::Float64
    r::Float64
    δ::Float64
end
function CreditWorkspace(m::DiscreteCreditModel,g::StateGrid,capital_points)
    nx,nz,na = discrete_dimensions(g)
    ax = getfield(g,:axes)
    zs = haskey(ax,:z) ? Float64.(ax.z) : [1.0]
    abilities = haskey(ax,:ability) ? Float64.(ax.ability) : [1.0]
    λ = leverage(m)
    top = λ*last(g.k)
    grid = collect(range(0.0,top;length=capital_points))
    b = m.model
    if b isa DiscreteSkibaModel
        # The technology switch and the offset are where the payoff bends.
        append!(grid,filter(x -> 0 < x < top,[b.κ,technology_crossing(b)]))
        sort!(unique!(grid))
    end
    output = [credit_output(m,k,zs[z],abilities[a]) for k in grid,z in 1:nz,a in 1:na]
    x = Float64.(g.k)
    return CreditWorkspace(x,vcat(0.0,x),grid,output,productivity_transition(m,g),zs,abilities,λ,m.r,preferences(m).δ)
end
function credit_output(m::DiscreteCreditModel,k,z,a)
    names = state_schema(m)
    s = NamedTuple{names}(map(n -> n == :k ? k : n == :z ? z : a,names))
    return production(m.model,s)
end

# Linear interpolation of a value column on the cash grid, clamped at both ends. The
# solver reports how often clamping occurs rather than extrapolating.
function cash_value(x,v,point)
    point <= first(x) && return first(v)
    point >= last(x) && return last(v)
    j = searchsortedlast(x,point)
    w = (point-x[j])/(x[j+1]-x[j])
    return (1-w)*v[j]+w*v[j+1]
end
next_cash(ws,y,k,s) = y-(ws.r+ws.δ)*k+(1+ws.r)*s

# Expected continuation value of capital k with own funds s, at current productivity z.
function expected_value(m,ws,V,s,k,z,a;output=nothing)
    total = 0.0
    for zp in axes(ws.P,2)
        p = ws.P[z,zp]
        p == 0 && continue
        y = output === nothing ? credit_output(m,k,ws.zs[zp],ws.abilities[a]) : output[zp]
        total += p*cash_value(ws.x,view(V,:,zp,a),next_cash(ws,y,k,s))
    end
    return total
end

# Allocation stage: best capital for own funds s. Grid search over the whole feasible
# interval finds the global maximum among grid points (and the limit λs), including across
# the Skiba bend. Ties keep lower capital. The Bellman iteration uses this fixed candidate
# set, so the operator is a contraction. With `refine`, golden-section search then improves
# the choice within the neighbouring cells; its trial points depend on V, so it is used only
# for reported off-grid choices, never inside the iteration.
function allocate(m,ws,V,s,z,a;refine=false)
    top = ws.λ*s
    best,best_k,best_j = -Inf,0.0,1
    for j in eachindex(ws.capital)
        k = ws.capital[j]
        k <= top || break
        q = expected_value(m,ws,V,s,k,z,a;output=view(ws.output,j,:,a))
        if q > best
            best,best_k,best_j = q,k,j
        end
    end
    # The limit itself is a candidate; it usually lies between capital grid points.
    feasible = searchsortedlast(ws.capital,top)
    q = expected_value(m,ws,V,s,top,z,a)
    q > best && ((best,best_k,best_j) = (q,top,feasible+1))
    # Refine within the neighbouring cells, never beyond the borrowing limit.
    refine || return (value=best,capital=best_k)
    lo = best_j > 1 ? min(ws.capital[best_j-1],top) : 0.0
    hi = best_j < feasible ? min(ws.capital[best_j+1],top) : top
    f(k) = expected_value(m,ws,V,s,k,z,a)
    ratio = (sqrt(5)-1)/2
    c1,c2 = hi-ratio*(hi-lo),lo+ratio*(hi-lo)
    f1,f2 = f(c1),f(c2)
    for _ in 1:40
        hi-lo <= 1e-10*max(1.0,hi) && break
        if f1 >= f2
            hi,c2,f2 = c2,c1,f1
            c1 = hi-ratio*(hi-lo); f1 = f(c1)
        else
            lo,c1,f1 = c1,c2,f2
            c2 = lo+ratio*(hi-lo); f2 = f(c2)
        end
    end
    for (k,q) in ((c1,f1),(c2,f2))
        if q > best && 0 <= k <= top
            best,best_k = q,k
        end
    end
    return (value=best,capital=best_k)
end

# One Bellman step. Allocation on the own-funds grid gives the continuation G(s); the
# saving stage chooses s for each cash state with the existing interval search.
function credit_step(m,ws,V,β,γ)
    nx,nz,na = size(V)
    G = zeros(length(ws.s),nz,na)
    nodes = zeros(length(ws.s),nz,na)
    for a in 1:na, z in 1:nz, j in eachindex(ws.s)
        q = allocate(m,ws,V,ws.s[j],z,a)
        G[j,z,a] = q.value
        nodes[j,z,a] = q.capital
    end
    all(isfinite,G) || throw(ArgumentError("Nonfinite continuation values"))
    out = similar(V)
    savings = similar(V)
    consumption = similar(V)
    for a in 1:na, z in 1:nz
        g = view(G,:,z,a)
        slopes = diff(g)./diff(ws.s)
        for i in 1:nx
            q = bellman_choice(ws.x[i],ws.s,g,slopes,β,γ)
            out[i,z,a] = q.value
            savings[i,z,a] = q.capital
            consumption[i,z,a] = q.consumption
        end
    end
    return (value=out,savings=savings,consumption=consumption,continuation=G,nodes=nodes)
end

# The saving stage treats the continuation as linear between own-funds nodes, so a
# choice between nodes is a lottery over them, each with its own capital. Policy
# evaluation, the transition matrix and market aggregates all use this lottery, so they
# match the Bellman step exactly. Off-grid queries re-optimize instead.
function own_funds_lottery(ws,s)
    j = clamp(searchsortedlast(ws.s,s),1,length(ws.s)-1)
    return j,(s-ws.s[j])/(ws.s[j+1]-ws.s[j])
end
function lottery_value(m,ws,V,s,nodes,z,a)
    j,w = own_funds_lottery(ws,s)
    lo = expected_value(m,ws,V,ws.s[j],nodes[j,z,a],z,a)
    w == 0 && return lo
    return (1-w)*lo+w*expected_value(m,ws,V,ws.s[j+1],nodes[j+1,z,a],z,a)
end

# Mean capital under the own-funds lottery.
function lottery_capital(ws,savings,nodes)
    nx,nz,na = size(savings)
    return [begin
        j,w = own_funds_lottery(ws,savings[i,z,a])
        (1-w)*nodes[j,z,a]+w*nodes[j+1,z,a]
    end for i in 1:nx,z in 1:nz,a in 1:na]
end
function policy_value(m,ws,V,savings,nodes,reward,β)
    nx,nz,na = size(V)
    return [reward[i,z,a]+β*lottery_value(m,ws,V,savings[i,z,a],nodes,z,a) for i in 1:nx,z in 1:nz,a in 1:na]
end

"""Solution of a [`DiscreteCreditModel`](@ref): value and policies on the cash-on-hand
grid, with diagnostics and a row-stochastic transition matrix. Policies are
`consumption`, `savings` (own funds), `next_capital` and `debt` (means over the
own-funds lottery), and the implied `leverage`."""
struct DiscreteCreditSolution{M,G,V,P,D,S,A,C}
    model::M
    grid::G
    value::V
    policy::P
    converged::Bool
    diagnostics::D
    settings::S
    initial_value::V
    transition::A
    continuation::C
    function DiscreteCreditSolution(::OwnedSnapshot,m,g,v,p,c,d,s,initial,a,continuation)
        new{typeof(m),typeof(g),typeof(v),typeof(p),typeof(d),typeof(s),typeof(a),typeof(continuation)}(
            m,g,v,p,c,d,s,initial,a,continuation)
    end
end
Base.propertynames(::DiscreteCreditSolution) = (:model,:grid,:value,:policy,:converged,:diagnostics,:settings,:initial_value,:transition)
function Base.getproperty(fit::DiscreteCreditSolution,key::Symbol)
    key == :continuation && throw(ArgumentError("Cached continuation values are private"))
    key == :model && return deepcopy(getfield(fit,key))
    return _readonly(getfield(fit,key))
end
function require_converged(fit::DiscreteCreditSolution)
    fit.converged || throw(ArgumentError("Credit Bellman solve did not converge: $(fit.diagnostics.stop_reason), residual=$(fit.diagnostics.residual_norm)"))
    return fit
end

"""
    solve(model::DiscreteCreditModel, grid; initial=nothing, maxiters=2000,
        value_tolerance=1e-8, residual_tolerance=1e-8,
        algorithm=:modified_policy_iteration, policy_steps=20, capital_points=nothing,
        history=false)

Two-stage Bellman solve on the cash-on-hand grid `grid.k`. Own funds are chosen on a
grid of zero followed by the cash grid; a choice between grid points is a lottery over
its neighbours, each with its own capital, which is what the transition matrix and
`policy.next_capital` use. `credit_policy` and `simulate` re-optimize at the exact
own funds instead. `capital_points` (default: twice the cash grid length) sets the fixed
capital grid for the allocation search. The Bellman iteration chooses on that grid, so it
is a contraction; reported off-grid choices are refined by golden-section search between
grid points. Next-period cash outside the grid is clamped to its ends; check
`diagnostics.boundary`, and widen the grid if either count is not small.

Modified policy iteration can cycle between two policies when a policy jump is nearly
tied. If its residual has not halved for 50 iterations, the solve continues with value
iteration; `diagnostics.switched_to_value_iteration` records the iteration (zero if not).
"""
const CREDIT_STALL_ITERATIONS = 50
function solve(m::DiscreteCreditModel,g::StateGrid; initial=nothing,maxiters=2000,
        value_tolerance=1e-8,residual_tolerance=1e-8,algorithm=:modified_policy_iteration,
        policy_steps=20,capital_points=nothing,history=false)
    validate_model_grid(m,g)
    all(x -> isfinite(x) && x > 0,g.k) || throw(ArgumentError("Cash-on-hand coordinates must be finite and positive"))
    maxiters isa Integer && maxiters >= 0 || throw(ArgumentError("maxiters must be a nonnegative integer"))
    policy_steps isa Integer && policy_steps >= 0 || throw(ArgumentError("policy_steps must be a nonnegative integer"))
    all(x -> isfinite(x) && x > 0,(value_tolerance,residual_tolerance)) || throw(ArgumentError("Tolerances must be finite and positive"))
    algorithm in (:value_iteration,:modified_policy_iteration) || throw(ArgumentError("Unknown Bellman algorithm"))
    points = capital_points === nothing ? 2length(g.k) : capital_points
    points isa Integer && points >= 2 || throw(ArgumentError("capital_points must be an integer of at least two"))
    g = deepcopy(g)
    raw,strategy = discrete_initial(initial,g)
    dims = discrete_dimensions(g)
    V = reshape(copy(raw),dims)
    initial_snapshot = copy(V)
    ws = CreditWorkspace(m,g,points)
    β,γ = preferences(m).discount,preferences(m).γ
    step = credit_step(m,ws,V,β,γ)
    trace = NamedTuple{(:iteration,:update,:residual),Tuple{Int,Float64,Float64}}[]
    update,residual,stop,iterations = Inf,maximum(abs,step.value.-V),:iteration_limit,0
    # Modified policy iteration can cycle between two policies when a policy jump is
    # nearly tied. If its residual stops improving, finish with value iteration, which is
    # a contraction from wherever it starts; the switch is reported in the diagnostics.
    mode = algorithm
    best_residual,stalled,switched_at = Inf,0,0
    for n in 1:maxiters
        iterations = n
        candidate = step.value
        if mode == :modified_policy_iteration
            reward = discrete_utility.(step.consumption,γ)
            for _ in 1:policy_steps
                candidate = policy_value(m,ws,candidate,step.savings,step.nodes,reward,β)
            end
        end
        all(isfinite,candidate) || (stop = :nonfinite; break)
        update = maximum(abs,candidate.-V)
        next = try
            credit_step(m,ws,candidate,β,γ)
        catch err
            err isa ArgumentError || rethrow()
            stop = :nonfinite; break
        end
        residual = maximum(abs,next.value.-candidate)
        isfinite(residual) || (stop = :nonfinite; break)
        V,step = candidate,next
        history && push!(trace,(iteration=n,update=update,residual=residual))
        if residual < 0.5*best_residual
            best_residual,stalled = residual,0
        else
            stalled += 1
        end
        if mode == :modified_policy_iteration && stalled >= CREDIT_STALL_ITERATIONS
            mode,switched_at = :value_iteration,n
        end
        if update <= value_tolerance && residual <= residual_tolerance
            stop = :converged; break
        end
    end
    capital = lottery_capital(ws,step.savings,step.nodes)
    debt = capital.-step.savings
    nx,nz,na = dims
    P,clamps,minimum_cash = credit_transition_matrix(m,ws,step.savings,step.nodes)
    budget = maximum(abs,step.consumption.+step.savings.-reshape(ws.x,nx,1,1))
    # The constraint holds at every node, so it holds for the lottery means as well.
    node_debt = step.nodes.-reshape(ws.s,:,1,1)
    collateral = maximum((1+ws.r).*node_debt.-m.θ*(1-ws.δ).*step.nodes)
    shaped(x) = reshape(copy(x),size(g))
    policy = (consumption=shaped(step.consumption),savings=shaped(step.savings),next_capital=shaped(capital),
        debt=shaped(debt),leverage=ws.λ)
    diagnostics = (stop_reason=stop,iterations=iterations,update_norm=update,residual_norm=residual,
        value_error_bound=residual/(1-β),budget_error=budget,collateral_slack=-collateral,
        switched_to_value_iteration=switched_at,
        minimum_next_cash=minimum_cash,boundary=(lower_cash=clamps.lower,upper_cash=clamps.upper,
        zero_savings=shaped(step.savings .== 0)),history=copy(trace))
    settings = (;algorithm,policy_steps,maxiters,value_tolerance,residual_tolerance,
        capital_points=points,initialization=strategy,history,precision=:Float64)
    return DiscreteCreditSolution(OWNED,deepcopy(m),g,shaped(V),policy,stop == :converged,diagnostics,
        settings,shaped(initial_snapshot),P,(G=step.continuation,V=copy(V),nodes=step.nodes,workspace=ws))
end
# Transition over the own-funds lottery: each node carries its own capital and next cash.
function credit_transition_matrix(m,ws,savings,nodes)
    nx,nz,na = size(savings)
    n = nx*nz*na
    rows,cols,weights = Int[],Int[],Float64[]
    lower = upper = 0
    smallest = Inf
    for a in 1:na, z in 1:nz, i in 1:nx
        index = i+nx*(z-1)+nx*nz*(a-1)
        j,w = own_funds_lottery(ws,savings[i,z,a])
        for (node,chance) in ((j,1-w),(j+1,w))
            chance > 0 || continue
            k,own = nodes[node,z,a],ws.s[node]
            for zp in 1:nz
                p = ws.P[z,zp]*chance
                p == 0 && continue
                x = next_cash(ws,credit_output(m,k,ws.zs[zp],ws.abilities[a]),k,own)
                smallest = min(smallest,x)
                x < first(ws.x) && (lower += 1)
                x > last(ws.x) && (upper += 1)
                x = clamp(x,first(ws.x),last(ws.x))
                c = clamp(searchsortedlast(ws.x,x),1,nx-1)
                v = (x-ws.x[c])/(ws.x[c+1]-ws.x[c])
                offset = nx*(zp-1)+nx*nz*(a-1)
                v < 1 && (push!(rows,index); push!(cols,c+offset); push!(weights,(1-v)*p))
                v > 0 && (push!(rows,index); push!(cols,c+1+offset); push!(weights,v*p))
            end
        end
    end
    return sparse(rows,cols,weights,n,n),(lower=lower,upper=upper),smallest
end

"""
    credit_policy(fit::DiscreteCreditSolution, state)

Choices at `state`, whose `k` is cash on hand: `consumption`, `savings` (own funds),
`capital` (next-period capital), `debt`, and `value`. Re-optimized against the fitted
continuation at the query state, not interpolated across policy jumps.
"""
function credit_policy(fit::DiscreteCreditSolution,state::NamedTuple)
    require_converged(fit)
    z,a = discrete_state_indices(fit.grid,state)
    cache = getfield(fit,:continuation)
    ws = cache.workspace
    m = getfield(fit,:model)
    p = preferences(m)
    g = view(cache.G,:,z,a)
    q = bellman_choice(state.k,ws.s,g,diff(g)./diff(ws.s),p.discount,p.γ)
    capital = allocate(m,ws,cache.V,q.capital,z,a;refine=true).capital
    return (consumption=q.consumption,savings=q.capital,capital=capital,debt=capital-q.capital,value=q.value)
end
next_capital(fit::DiscreteCreditSolution,state::NamedTuple) = credit_policy(fit,state).capital
consumption(fit::DiscreteCreditSolution,state::NamedTuple) = credit_policy(fit,state).consumption

"""
    simulate(fit::DiscreteCreditSolution, initial_state, periods; rng=Random.default_rng())

One household's path, with cash on hand continuous and productivity drawn from the
solved chain. `initial_state.k` is initial cash on hand. Returns `times`, `states`
(cash on hand, productivity, ability; length periods+1), and per-period choices of
length `periods`: `consumption`, `savings`, `capital` (bought in period t and held in
t+1), `debt`, and `output` (produced in t+1 by that capital at the next productivity).
Survey assets in period t+1 are `capital[t]`, which is set before that period's shock.
"""
function simulate(fit::DiscreteCreditSolution,initial::NamedTuple,periods;rng=Random.default_rng())
    require_converged(fit)
    n = discrete_period(periods)
    discrete_state_indices(fit.grid,initial)
    names = keys(getfield(fit.grid,:axes))
    state = NamedTuple{names}(map(key -> Float64(initial[key]),names))
    m = getfield(fit,:model)
    ws = getfield(fit,:continuation).workspace
    states = Vector{typeof(state)}(undef,n+1); states[1] = state
    out = (consumption=zeros(n),savings=zeros(n),capital=zeros(n),debt=zeros(n),output=zeros(n))
    for t in 1:n
        choice = credit_policy(fit,state)
        z,a = discrete_state_indices(fit.grid,state)
        zp = z
        if :z in names
            draw = rand(rng); cumulative = 0.0; zp = size(ws.P,2)
            for j in axes(ws.P,2)
                cumulative += ws.P[z,j]
                draw < cumulative && (zp = j; break)
            end
        end
        y = credit_output(m,choice.capital,ws.zs[zp],ws.abilities[a])
        x = next_cash(ws,y,choice.capital,choice.savings)
        # Clamping keeps queries on the solved grid; it is reported by the solve diagnostics.
        x = clamp(x,first(ws.x),last(ws.x))
        out.consumption[t] = choice.consumption; out.savings[t] = choice.savings
        out.capital[t] = choice.capital; out.debt[t] = choice.debt; out.output[t] = y
        state = merge(state,(k=x,))
        :z in names && (state = merge(state,(z=ws.zs[zp],)))
        states[t+1] = state
    end
    return merge((times=collect(0:n),states=states),out)
end

Evolution(fit::DiscreteCreditSolution;kwargs...) = (isempty(kwargs) ||
    throw(ArgumentError("Discrete evolution uses integer periods and takes no dt"));
    require_converged(fit); DiscreteEvolution(OWNED,deepcopy(fit.grid),copy(getfield(fit,:transition))))

"""
    stationary(fit::DiscreteCreditSolution, initial=nothing; require_aperiodic=false)

Invariant probabilities on the cash-on-hand grid, as for the base discrete models.
"""
function stationary(fit::DiscreteCreditSolution,initial::Union{Nothing,StateDistribution}=nothing;require_aperiodic=false)
    require_converged(fit)
    P = getfield(fit,:transition)
    cash = vec(coordinates(fit.grid).k)
    if initial !== nothing
        compatible_grid(fit.grid,initial.grid) || throw(ArgumentError("Initial grid differs from fit"))
        p = stationary_transition(P;initial_mass=vec(mass(initial)),require_aperiodic)
    elseif population(getfield(fit,:model)) === nothing
        p = stationary_transition(P;require_aperiodic,capital=cash)
    else
        nx,nz,na = discrete_dimensions(fit.grid)
        types = population(getfield(fit,:model))
        block = nx*nz
        p = zeros(length(fit.grid))
        for a in 1:na
            types.shares[a] == 0 && continue
            r = (a-1)*block+1:a*block
            p[r] .= types.shares[a].*stationary_transition(P[r,r];require_aperiodic,capital=cash[r])
        end
    end
    return distribution(fit.grid;masses=reshape(p,size(fit.grid)))
end

"""Outcome of [`discrete_credit_equilibrium`](@ref): `scan`, `equilibria` (with `fit` and
`distribution`), and unresolved `discontinuities`, as for [`CreditEquilibrium`](@ref)."""
struct DiscreteCreditEquilibrium{M,E,D,S} <: AbstractCreditEquilibrium
    model::M
    θ::Float64
    scan::Vector{S}
    equilibria::E
    discontinuities::D
    settings::NamedTuple
    DiscreteCreditEquilibrium(::OwnedSnapshot,m,θ,scan::Vector{S},e,d,settings) where S =
        new{typeof(m),typeof(e),typeof(d),S}(m,θ,scan,e,d,settings)
end
Base.propertynames(::DiscreteCreditEquilibrium) = (:model,:θ,:scan,:equilibria,:discontinuities,:settings)
function Base.getproperty(eq::DiscreteCreditEquilibrium,key::Symbol)
    key == :model && return deepcopy(getfield(eq,:model))
    value = getfield(eq,key)
    return value isa AbstractArray ? _readonly(value) : value
end
function show(io::IO,eq::DiscreteCreditEquilibrium)
    rates = [round(e.rate;sigdigits=5) for e in getfield(eq,:equilibria)]
    print(io,"DiscreteCreditEquilibrium(θ=",getfield(eq,:θ),", clearing rates=",rates,
        ", discontinuities=",length(getfield(eq,:discontinuities)),")")
end

function discrete_credit_state(base,grid,θ,r,initial,warm,tolerances,options,horizon)
    m = DiscreteCreditModel(base;θ=θ,r=r)
    fit = solve(m,grid;initial=warm,options...)
    !fit.converged && warm !== nothing && (fit = solve(m,grid;options...))
    failed(status,detail) = (summary=(rate=r,status=status,detail=detail,excess=NaN,relative_excess=NaN,
        capital=NaN,own_funds=NaN,borrowing=NaN,output=NaN,upper_mass=NaN,boundary_ok=false,late_change=NaN),
        fit=fit,distribution=nothing)
    fit.converged || return failed(:not_converged,string(fit.diagnostics.stop_reason))
    late_change = NaN
    d = try
        if horizon === nothing
            stationary(fit,initial)
        else
            e = Evolution(fit)
            path = trajectory(e,initial,[horizon-CREDIT_DISCRETE_WINDOW,horizon])
            late_change = sum(abs,view(path.probabilities,:,2).-view(path.probabilities,:,1))/2
            final(path)
        end
    catch err
        err isa ArgumentError || rethrow()
        return failed(:no_stationary,sprint(showerror,err))
    end
    p = vec(mass(d))
    dims = discrete_dimensions(grid)
    policy = map(x -> x isa AbstractArray ? reshape(x,dims) : x,getfield(fit,:policy))
    capital = sum(p.*vec(policy.next_capital))
    own = sum(p.*vec(policy.savings))
    borrowing = sum(p.*max.(vec(policy.debt),0.0))
    # Expected output next period from the capital chosen now.
    ws = getfield(fit,:continuation).workspace
    nx,nz,na = discrete_dimensions(grid)
    output = 0.0
    nodes = getfield(fit,:continuation).nodes
    for a in 1:na, z in 1:nz, i in 1:nx
        q = p[i+nx*(z-1)+nx*nz*(a-1)]
        q == 0 && continue
        j,w = own_funds_lottery(ws,policy.savings[i,z,a])
        for (node,chance) in ((j,1-w),(j+1,w))
            chance > 0 || continue
            output += q*chance*sum(ws.P[z,zp]*credit_output(m,nodes[node,z,a],ws.zs[zp],ws.abilities[a]) for zp in 1:nz)
        end
    end
    upper = boundary_mass(d).upper
    summary = (rate=r,status=:solved,detail="",excess=capital-own,relative_excess=(capital-own)/own,
        capital=capital,own_funds=own,borrowing=borrowing,output=output,upper_mass=upper,
        boundary_ok=upper <= tolerances.boundary,late_change=late_change)
    return (summary=summary,fit=fit,distribution=d)
end
const CREDIT_DISCRETE_WINDOW = 10

"""
    discrete_credit_equilibrium(model, grid; θ, rates=nothing, initial=nothing, horizon=nothing,
        rate_tolerance=1e-5, clearing_tolerance=1e-3, boundary_tolerance=1e-4,
        solve_options=NamedTuple())

Rates at which household debt nets to zero, `∫(k′−s)dμ = 0`, for the discrete credit
model with recovery share `θ`. By default `rates` scans `(−δ, 1/discount − 1)`. The
market clears on the stationary distribution, or with an integer `horizon` and an
explicit `initial` on the distribution after that many periods at each fixed rate.
Multiple and unresolved crossings are reported as for [`credit_equilibrium`](@ref).
Requires productivity risk, so that some households lend below `1/discount − 1`.
"""
function discrete_credit_equilibrium(base::Union{DiscreteSkibaModel,DiscreteRamseyCassKoopmansModel},grid::StateGrid;
        θ,rates=nothing,initial=nothing,horizon=nothing,rate_tolerance=1e-5,clearing_tolerance=1e-3,
        boundary_tolerance=1e-4,solve_options=NamedTuple())
    process(base) === nothing && throw(ArgumentError("discrete_credit_equilibrium requires productivity risk"))
    0 <= θ <= 1 || throw(ArgumentError("θ must lie between zero and one"))
    haskey(solve_options,:initial) && throw(ArgumentError("The outer loop manages warm starts; do not pass initial in solve_options"))
    if horizon !== nothing
        initial === nothing && throw(ArgumentError("A finite horizon needs an explicit initial distribution"))
        horizon isa Integer && horizon > CREDIT_DISCRETE_WINDOW || throw(ArgumentError("horizon must be an integer above $CREDIT_DISCRETE_WINDOW"))
    end
    initial === nothing || compatible_grid(grid,initial.grid) || throw(ArgumentError("Initial distribution grid differs from grid"))
    δ,β = base.depreciation,base.discount
    top = 1/β-1
    width = top+δ
    scan_rates = rates === nothing ? collect(range(-δ+.05width,top-.02width;length=13)) : sort!(float.(collect(rates)))
    length(scan_rates) >= 2 && all(isfinite,scan_rates) && allunique(scan_rates) ||
        throw(ArgumentError("rates must contain at least two distinct finite values"))
    all(r -> -δ < r < top,scan_rates) || throw(ArgumentError("rates must lie strictly between -δ and 1/discount-1"))
    tolerances = (rate=rate_tolerance,clearing=clearing_tolerance,boundary=boundary_tolerance)
    evaluate(r,warm) = discrete_credit_state(base,grid,θ,r,initial,warm,tolerances,solve_options,horizon)
    scan,equilibria,discontinuities = clear_credit_market(evaluate,scan_rates,tolerances)
    settings = (rate_tolerance=rate_tolerance,clearing_tolerance=clearing_tolerance,boundary_tolerance=boundary_tolerance,
        rates=copy(scan_rates),long_run=horizon === nothing ? :stationary : :horizon,horizon=horizon)
    return DiscreteCreditEquilibrium(OWNED,deepcopy(base),Float64(θ),scan,equilibria,discontinuities,settings)
end
