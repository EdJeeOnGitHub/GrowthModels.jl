"""Integer-period propagation of individual-risk distributions. Construct with
`Evolution(fit::DiscreteGrowthSolution)`; no integration timestep is needed."""
struct DiscreteEvolution{G,P}
    grid::G
    transition::P
    function DiscreteEvolution(::OwnedSnapshot,g,p)
        new{typeof(g),typeof(p)}(g,p)
    end
end
Base.propertynames(::DiscreteEvolution) = (:grid,)
function Base.getproperty(e::DiscreteEvolution,key::Symbol)
    key == :grid || throw(ArgumentError("DiscreteEvolution exposes only grid; transition storage is private"))
    return getfield(e,key)
end
function Evolution(fit::DiscreteGrowthSolution;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Discrete evolution uses integer periods and takes no dt"))
    require_converged(fit)
    return DiscreteEvolution(OWNED,deepcopy(fit.grid),copy(getfield(fit,:transition)))
end
function discrete_period(x)
    x isa Integer && !(x isa Bool) && 0 <= x <= typemax(Int) ||
        throw(ArgumentError("Discrete periods must be nonnegative integers representable as Int"))
    return Int(x)
end
function discrete_times(times,t0)
    origin=discrete_period(t0)
    ts=discrete_period.(collect(times))
    !isempty(ts) && first(ts)>=origin && all(diff(ts) .> 0) ||
        throw(ArgumentError("Observation periods must be nonempty, increasing, and at or after t0"))
    return ts
end
function walk_discrete(emit,e::DiscreteEvolution,initial,times,t0)
    compatible_grid(e.grid,initial.grid) || throw(ArgumentError("Initial grid differs from evolution grid"))
    p=Float64.(vec(mass(initial)))
    validate_probability(p)
    out=similar(p)
    forward=transpose(getfield(e,:transition))
    current=t0
    max_error=abs(sum(p)-1)
    for (i,t) in enumerate(times)
        while current<t
            mul!(out,forward,p)
            validate_probability(out)
            max_error=max(max_error,abs(sum(out)-1))
            p,out=out,p
            current+=1
        end
        emit(i,p)
    end
    return (steps=current-t0,max_mass_error=max_error)
end
function trajectory(e::DiscreteEvolution,initial::StateDistribution,times;t0=0)
    ts=discrete_times(times,t0)
    saved=Matrix{Float64}(undef,length(e.grid),length(ts))
    diagnostic=walk_discrete(e,initial,ts,t0) do i,p
        saved[:,i].=p
    end
    return Trajectory(OWNED,deepcopy(e.grid),ts,saved,
        (t0=t0,algorithm=:discrete_transition,interpolation=:none),diagnostic)
end
function advance(e::DiscreteEvolution,initial::StateDistribution,periods;t0=0)
    n,origin=discrete_period(periods),discrete_period(t0)
    n <= typemax(Int)-origin || throw(ArgumentError("Final period overflows Int"))
    saved=zeros(length(e.grid))
    walk_discrete(e,initial,[origin+n],origin) do _,p
        copyto!(saved,p)
    end
    return distribution(e.grid;masses=reshape(saved,size(e.grid)))
end
function observe(e::DiscreteEvolution,initial::StateDistribution,times,observables::NamedTuple;t0=0)
    isempty(observables) && throw(ArgumentError("Supply at least one observable"))
    ts=discrete_times(times,t0)
    weights=map(f -> vec([f(grid_state(e.grid,i)) for i in CartesianIndices(size(e.grid))]),observables)
    result=map(w -> Vector{promote_type(Float64,eltype(w))}(undef,length(ts)),weights)
    diagnostic=walk_discrete(e,initial,ts,t0) do j,p
        for key in keys(weights)
            result[key][j]=sum(weights[key].*p)
        end
    end
    return Observations(OWNED,ts,result,
        (t0=t0,algorithm=:discrete_transition,interpolation=:none),diagnostic)
end

# For a strongly connected class, gcd of edge distance differences is its period.
function communicating_period(Pt,states)
    members=falses(size(Pt,1)); members[states].=true
    distance=fill(-1,size(Pt,1)); distance[first(states)]=0
    queue=[first(states)]
    rows,weights=rowvals(Pt),nonzeros(Pt)
    cursor=1
    while cursor<=length(queue)
        source=queue[cursor]; cursor+=1
        for ptr in nzrange(Pt,source)
            dest=rows[ptr]
            if weights[ptr]>0 && members[dest] && distance[dest]<0
                distance[dest]=distance[source]+1
                push!(queue,dest)
            end
        end
    end
    period=0
    for source in states, ptr in nzrange(Pt,source)
        dest=rows[ptr]
        weights[ptr]>0 && members[dest] || continue
        period=gcd(period,abs(distance[source]+1-distance[dest]))
    end
    return period
end

# Invariant/Cesaro probabilities. This works directly with I-P′ and reuses only
# the graph and factorization helpers from continuous time, not its interpretation.
function stationary_transition(P;initial_mass=nothing,require_aperiodic=false)
    check_transition(P)
    Pt=sparse(transpose(P))
    classes,transient=closed_communicating_classes(Pt)
    n=size(P,1)
    if initial_mass === nothing
        length(classes)==1 || throw(ArgumentError("Multiple stationary classes: supply an initial distribution"))
    else
        length(initial_mass)==n || throw(DimensionMismatch("Initial probabilities differ from transition size"))
        validate_probability(initial_mass)
    end
    occupation=initial_mass !== nothing && !isempty(transient) ?
        (I-Pt[transient,transient]) \ initial_mass[transient] : nothing
    result=zeros(n)
    for states in classes
        weight=initial_mass === nothing ? 1.0 : sum(initial_mass[states])+
            (occupation === nothing ? 0.0 : sum(Pt[states,transient]*occupation))
        weight == 0 && continue
        if require_aperiodic && communicating_period(Pt,states)!=1
            throw(ArgumentError("A positive-mass recurrent class is periodic; stationary returns time-average probabilities, not necessarily a pointwise limit"))
        end
        system=sparse(I-Pt[states,states])
        system[1,:].=0
        system[1,1]=1
        rhs=zeros(length(states)); rhs[1]=1
        invariant=system \ rhs
        invariant ./= sum(invariant)
        result[states].=weight.*invariant
    end
    validate_probability(result)
    return result
end

"""
    stationary(fit::DiscreteGrowthSolution, initial=nothing; require_aperiodic=false)

Invariant probabilities, with recurrent-class weights selected by initial
probabilities and transient absorption. They equal long-run time-average
probabilities even for periodic chains; pointwise convergence is not promised.
Set `require_aperiodic=true` to reject periodic recurrent classes carrying mass.
Without initial probabilities, population shares weight ability blocks only when
each positive-share block has a unique closed class. Explicit initial probabilities
override population shares. A unique productivity invariant law alone does not
imply uniqueness of the joint capital/productivity invariant law.
"""
function stationary(fit::DiscreteGrowthSolution,initial::Union{Nothing,StateDistribution}=nothing;
        require_aperiodic=false)
    require_converged(fit)
    P=getfield(fit,:transition)
    if initial !== nothing
        compatible_grid(fit.grid,initial.grid) || throw(ArgumentError("Initial grid differs from fit"))
        p=stationary_transition(P;initial_mass=vec(mass(initial)),require_aperiodic)
    else
        types=population(getfield(fit,:model))
        if types === nothing
            p=stationary_transition(P;require_aperiodic)
        else
            nk,nz,na=discrete_dimensions(fit.grid)
            blocksize=nk*nz
            p=zeros(length(fit.grid))
            for a in 1:na
                share=types.shares[a]
                share == 0 && continue
                indices=(a-1)*blocksize+1:a*blocksize
                p[indices].=share.*stationary_transition(P[indices,indices];require_aperiodic)
            end
        end
    end
    return distribution(fit.grid;masses=reshape(p,size(fit.grid)))
end
