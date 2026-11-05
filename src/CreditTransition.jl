# Perfect-foresight paths of the credit market after an unanticipated change in λ.
#
# Each outer iteration solves the household problem backward from the terminal value
# and pushes the distribution forward, with implicit steps as in Achdou et al. Rates
# then update as in Buera and Shin (2013): at each date, find the rate that clears the
# static market given that date's distribution, and damp toward it. Capital demand
# depends on the rate only through the static choice, which makes this step cheap.
#
# Jumps in demand at technology switches mean a date's market may not clear exactly.
# Convergence is therefore judged on the rate, and the remaining clearing error is
# reported for checking against refinement of the productivity grid.

"""Equilibrium path from [`credit_transition`](@ref). `times` holds the `N` dates
`0, dt, ..., T-dt` with `rates`, `excess`, `capital`, `wealth`, `borrowing`, `output`,
and `high_technology` at each. `distributions` is a `Trajectory` with one more date, at
`T`. `value` is the value function at `t=0`, `terminal` the terminal equilibrium, and
`diagnostics` reports convergence, clearing errors, and the distance from the terminal
stationary distribution at `T`."""
struct CreditTransition{M,G,T,E,D}
    model::M
    λ::Float64
    grid::G
    times::Vector{Float64}
    rates::Vector{Float64}
    excess::Vector{Float64}
    capital::Vector{Float64}
    wealth::Vector{Float64}
    borrowing::Vector{Float64}
    output::Vector{Float64}
    high_technology::Vector{Float64}
    value::Array{Float64}
    distributions::T
    terminal::E
    converged::Bool
    diagnostics::D
    CreditTransition(::OwnedSnapshot,m,λ,g,t,r,e,k,a,b,y,h,v,paths::P,terminal::E,c,d) where {P,E} =
        new{typeof(m),typeof(g),P,E,typeof(d)}(m,λ,g,t,r,e,k,a,b,y,h,v,paths,terminal,c,d)
end
Base.propertynames(::CreditTransition) = fieldnames(CreditTransition)
function Base.getproperty(path::CreditTransition,key::Symbol)
    key == :model && return deepcopy(getfield(path,:model))
    return _readonly(getfield(path,key))
end
function show(io::IO,path::CreditTransition)
    d = getfield(path,:diagnostics)
    print(io,"CreditTransition(λ=",getfield(path,:λ),", converged=",getfield(path,:converged),
        ", iterations=",d.iterations,", largest clearing error=",round(d.largest_relative_excess;sigdigits=3),")")
end
function require_converged(path::CreditTransition)
    d = getfield(path,:diagnostics)
    getfield(path,:converged) || throw(ArgumentError("Credit transition did not converge: largest rate change $(d.largest_rate_change) after $(d.iterations) iterations"))
    return path
end

# Reusable storage for one path. Only the rates change between passes, so factor
# patterns and drift storage persist across outer iterations.
struct CreditPathWorkspace{B,W,St,S1,S2}
    base::B
    λ::Float64
    dt::Float64
    ws::W
    states::St
    terminal_value::Array{Float64}
    drift::Matrix{Float64}
    masses::Matrix{Float64}
    forward::SparseMatrixCSC{Float64,Int}
    forward_map::Vector{Int}
    forward_diagonal::Vector{Int}
    backward_solver::S1
    forward_solver::S2
end
credit_model_at(base,λ,r) = CreditModel{Float64,typeof(base)}(OWNED,base,λ,r)

function CreditPathWorkspace(base,grid,λ,dt,N,terminal_rate,terminal_value,initial_mass)
    m = credit_model_at(base,λ,terminal_rate)
    ws = HJBWorkspace(m,grid)
    states = vec([grid_state(grid,i) for i in CartesianIndices(size(grid))])
    prepare_hjb_step!(ws,base,terminal_value,dt)
    n = length(grid)
    # Forward systems are I-dt*A′. A′ has a fixed pattern, and these maps fill it in place.
    labels = SparseMatrixCSC(n,n,copy(ws.A.colptr),copy(rowvals(ws.A)),Float64.(1:nnz(ws.A)))
    transposed = sparse(transpose(labels))
    forward_map = Int.(nonzeros(transposed))
    forward = SparseMatrixCSC(n,n,copy(transposed.colptr),copy(rowvals(transposed)),zeros(nnz(transposed)))
    forward_diagonal = [first(p for p in nzrange(forward,j) if rowvals(forward)[p] == j) for j in 1:n]
    fill_forward_system!(forward,forward_map,forward_diagonal,ws,dt)
    blocksize = independent_blocksize(forward,preferred_blocksize(grid))
    masses = zeros(n,N+1)
    masses[:,1] .= initial_mass
    return CreditPathWorkspace(base,Float64(λ),Float64(dt),ws,states,copy(terminal_value),zeros(n,N),masses,
        forward,forward_map,forward_diagonal,SparseBlockSolver(ws.B;blocksize=blocksize),
        SparseBlockSolver(forward;blocksize=blocksize))
end

function fill_forward_system!(forward,forward_map,forward_diagonal,ws,dt)
    rates = nonzeros(ws.A)
    values = nonzeros(forward)
    for p in eachindex(values)
        values[p] = -dt*rates[forward_map[p]]
    end
    for p in forward_diagonal
        values[p] += 1
    end
    return forward
end

# Sets the generator from a stored drift, using the same upwinding as the HJB step:
# a positive drift moves mass up the wealth grid and a negative one moves it down.
function set_generator_from_drift!(ws,drift)
    Nk = size(ws.y,1)
    rates = nonzeros(ws.A)
    copyto!(rates,ws.diffusion_values)
    for offset in 0:Nk:length(drift)-1, k in 1:Nk
        i = offset+k
        s = drift[i]
        down = s < 0 ? -s/ws.dkb[k] : 0.0
        up = s > 0 ? s/ws.dkf[k] : 0.0
        rates[ws.diagonal[i]] -= down+up
        k > 1 && (rates[ws.backward[i]] += down)
        k < Nk && (rates[ws.forward[i]] += up)
    end
    return ws
end

function set_income!(ws,base,λ,r,states)
    m = credit_model_at(base,λ,r)
    δ,γ = preferences(base).δ,preferences(base).γ
    for i in eachindex(states)
        s = states[i]
        y = resources(m,s)
        ws.y[i] = y
        ws.c0[i] = y-δ*s.k
        ws.dV0[i] = max(ws.c0[i],eps())^(-γ)
    end
    return ws
end

# Backward from the terminal value. Income and policy at date n use r(n) and the value
# at date n+1, then the value at date n solves the implicit step. Returns the value at t=0.
function backward_pass!(cache,rates)
    ws = cache.ws
    v = copy(cache.terminal_value)
    next = similar(v)
    for n in length(rates):-1:1
        set_income!(ws,cache.base,cache.λ,rates[n],cache.states)
        prepare_hjb_step!(ws,cache.base,v,cache.dt)
        view(cache.drift,:,n) .= vec(ws.c0).-vec(ws.c)
        refactor!(cache.backward_solver,ws.B)
        solve_linear!(vec(next),cache.backward_solver,ws.rhs)
        all(isfinite,next) || throw(ValueFunctionError("Nonfinite value on the transition path"))
        copyto!(v,next)
    end
    return v
end

function forward_pass!(cache)
    N = size(cache.drift,2)
    for n in 1:N
        set_generator_from_drift!(cache.ws,view(cache.drift,:,n))
        fill_forward_system!(cache.forward,cache.forward_map,cache.forward_diagonal,cache.ws,cache.dt)
        refactor!(cache.forward_solver,cache.forward)
        solve_linear!(view(cache.masses,:,n+1),cache.forward_solver,view(cache.masses,:,n))
    end
    return cache.masses
end

function credit_aggregates(base,λ,r,states,p)
    m = credit_model_at(base,λ,r)
    cost = r+preferences(base).δ
    capital = wealth = borrowing = output = high = 0.0
    for i in eachindex(p)
        p[i] > 0 || continue
        s = states[i]
        choice = capital_choice(m,s)
        capital += p[i]*choice.capital
        wealth += p[i]*s.k
        borrowing += p[i]*max(choice.capital-s.k,0.0)
        output += p[i]*(choice.profit+cost*choice.capital)
        high += p[i]*(choice.technology > 1)
    end
    return (capital=capital,wealth=wealth,borrowing=borrowing,output=output,high_technology=high)
end
function capital_demand(base,λ,r,states,p)
    m = credit_model_at(base,λ,r)
    total = 0.0
    for i in eachindex(p)
        p[i] > 0 && (total += p[i]*capital_choice(m,states[i]).capital)
    end
    return total
end

const CREDIT_RATE_PRECISION = 1e-8

"""The rate clearing the static market for a given distribution. Each household's
capital falls weakly with the rate, so demand is monotone. At a jump this converges to
the switching rate, where the market cannot clear exactly."""
function static_clearing_rate(base,λ,states,p,wealth,guess)
    floor_rate = -preferences(base).δ+CREDIT_RATE_PRECISION
    excess(r) = capital_demand(base,λ,r,states,p)-wealth
    step = 1e-3
    lo = max(guess-step,floor_rate)
    while excess(lo) < 0 && lo > floor_rate
        step *= 2
        lo = max(guess-step,floor_rate)
    end
    step = 1e-3
    hi = guess+step
    while excess(hi) >= 0
        step *= 2
        hi = guess+step
        isfinite(hi) || throw(ArgumentError("No finite rate clears the static credit market"))
    end
    while hi-lo > CREDIT_RATE_PRECISION
        mid = lo+(hi-lo)/2
        excess(mid) >= 0 ? (lo = mid) : (hi = mid)
    end
    return lo+(hi-lo)/2
end

"""
    credit_transition(model, grid, initial; λ, T=100.0, dt=0.25, terminal=nothing,
        initial_rates=nothing, damping=0.3, rate_tolerance=1e-5, maxiters=300,
        boundary_tolerance=1e-4)

The perfect-foresight path after an unanticipated change to collateral limit `λ` at
`t=0`, from the distribution `initial` to the stationary equilibrium at `λ`. `terminal`
reuses a [`credit_equilibrium`](@ref) on the same grid. `initial_rates` is the starting
guess for `r(t)` on the `N` dates, and defaults to the terminal rate. Uniqueness is not
guaranteed, so compare different guesses and `damping` values. Wealth moves slowly, but capital
reallocates immediately at `t=0` because it is reversible and rented.

It iterates on the rate path `r(t)`: solve households backward from the terminal value,
push the distribution forward, find each date's static clearing rate, and move a share
`damping` toward it. It converges when no rate would change by more than
`rate_tolerance`, as in Buera and Shin (2013). Jumps in demand at technology switches can
leave a clearing error at some dates, so check `diagnostics.largest_relative_excess`
against refinement of `z`. Also check that `T` is long enough: `diagnostics.terminal`
reports the gap from the terminal equilibrium at `T`, which can stay large if escapes from
the trap are rare. Non-convergence returns the last iterate; use `require_converged`.
Automatic differentiation is not supported.
"""
function credit_transition(base::Model,grid::StateGrid,initial::StateDistribution; λ,T=100.0,dt=0.25,
        terminal=nothing,initial_rates=nothing,damping=0.3,rate_tolerance=1e-5,maxiters=300,boundary_tolerance=1e-4)
    credit_technologies(base)
    model_eltype(base) === Float64 || throw(ArgumentError("credit_transition requires Float64 model parameters; automatic differentiation is not supported"))
    compatible_grid(grid,initial.grid) || throw(ArgumentError("Initial distribution grid differs from grid"))
    all(x -> x isa Real && isfinite(x) && x > 0,(T,dt,rate_tolerance,boundary_tolerance)) ||
        throw(ArgumentError("T, dt, and tolerances must be finite and positive"))
    0 < damping <= 1 || throw(ArgumentError("damping must lie in (0, 1]"))
    maxiters isa Integer && maxiters >= 1 || throw(ArgumentError("maxiters must be a positive integer"))
    N = round(Int,T/dt)
    N >= 1 && isapprox(N*dt,T;atol=1e-8*T) || throw(ArgumentError("T must be a whole number of dt steps"))
    if terminal === nothing
        terminal = credit_equilibrium(base,grid;λ=λ,boundary_tolerance=boundary_tolerance)
    else
        terminal isa CreditEquilibrium && terminal.λ == λ || throw(ArgumentError("terminal must be a CreditEquilibrium with the same λ"))
    end
    terminal_rate = clearing_rate(terminal)
    terminal_state = only(terminal.equilibria)
    compatible_grid(grid,terminal_state.fit.grid) || throw(ArgumentError("Terminal equilibrium grid differs from grid"))

    terminal_value = Array{Float64}(copy(terminal_state.fit.value))
    start = vec(Array{Float64}(mass(initial)))
    cache = CreditPathWorkspace(base,grid,λ,dt,N,terminal_rate,terminal_value,start)
    rates = initial_rates === nothing ? fill(float(terminal_rate),N) : Float64.(collect(initial_rates))
    length(rates) == N && all(r -> isfinite(r) && r > -preferences(base).δ,rates) ||
        throw(ArgumentError("initial_rates must hold $N finite rates above -δ"))
    history = NamedTuple{(:iteration,:rate_change,:relative_excess),Tuple{Int,Float64,Float64}}[]
    value = terminal_value
    rate_change = Inf
    largest_excess = Inf
    converged = false
    iterations = 0
    targets = similar(rates)
    for iteration in 1:maxiters
        iterations = iteration
        value = backward_pass!(cache,rates)
        forward_pass!(cache)
        largest_excess = 0.0
        for n in 1:N
            p = view(cache.masses,:,n)
            wealth = sum(p[i]*cache.states[i].k for i in eachindex(p))
            largest_excess = max(largest_excess,abs(capital_demand(base,λ,rates[n],cache.states,p)-wealth)/wealth)
            targets[n] = static_clearing_rate(base,λ,cache.states,p,wealth,rates[n])
        end
        rate_change = maximum(abs,targets.-rates)
        push!(history,(iteration=iteration,rate_change=rate_change,relative_excess=largest_excess))
        if rate_change <= rate_tolerance
            converged = true
            break
        end
        rates .+= damping.*(targets.-rates)
    end

    times = collect((0:N-1).*dt)
    aggregates = [credit_aggregates(base,λ,rates[n],cache.states,view(cache.masses,:,n)) for n in 1:N]
    wealth = [a.wealth for a in aggregates]
    capital = [a.capital for a in aggregates]
    masses = copy(cache.masses)
    upper = [sum(selectdim(reshape(view(masses,:,n),size(grid)),1,size(grid)[1])) for n in 1:N+1]
    target = vec(Array(mass(terminal_state.distribution)))
    final_wealth = sum(masses[i,end]*cache.states[i].k for i in axes(masses,1))
    gap = (rate=last(rates)-terminal_rate,wealth=final_wealth-terminal_state.wealth,
        total_variation=sum(abs,view(masses,:,N+1).-target)/2)
    diagnostics = (iterations=iterations,largest_rate_change=rate_change,
        largest_relative_excess=largest_excess,history=history,terminal=gap,
        upper_mass=maximum(upper),boundary_ok=maximum(upper) <= boundary_tolerance,
        mass_error=maximum(abs,sum(masses;dims=1).-1))
    settings = (T=float(T),dt=float(dt),damping=damping,rate_tolerance=rate_tolerance,maxiters=maxiters)
    distributions = Trajectory(OWNED,deepcopy(grid),collect((0:N).*dt),masses,settings,(;))
    return CreditTransition(OWNED,deepcopy(base),Float64(λ),deepcopy(grid),times,rates,capital.-wealth,capital,wealth,
        [a.borrowing for a in aggregates],[a.output for a in aggregates],[a.high_technology for a in aggregates],
        copy(value),distributions,terminal,converged,diagnostics)
end
