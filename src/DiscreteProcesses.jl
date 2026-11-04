"""Native per-period productivity process. It supplies transition probabilities,
not a continuous-time generator."""
abstract type DiscreteProductivity <: StochasticProcess end

function check_transition(P)
    P isa AbstractMatrix || throw(ArgumentError("Transition probabilities require a matrix"))
    size(P,1) == size(P,2) > 0 || throw(ArgumentError("Transition matrix must be nonempty and square"))
    all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x) && 0 <= x <= 1,P) ||
        throw(ArgumentError("Transition probabilities must be finite and between zero and one"))
    maximum(abs, sum(P;dims=2) .- 1) <= 1e-12 ||
        throw(ArgumentError("Transition rows must sum to one; inputs are not normalized"))
    return P
end

"""
    MarkovChain(; values, transition)

Finite positive productivity levels and per-period probabilities. Row `i` gives
next-period probabilities conditional on current state `i`. Values must be
strictly increasing. Inputs are copied, validated, and never normalized.
"""
struct MarkovChain <: DiscreteProductivity
    values::Vector{Float64}
    transition::Matrix{Float64}
    function MarkovChain(;values,transition)
        all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x) && x > 0,values) ||
            throw(ArgumentError("Productivity values must be finite, positive, and non-AD"))
        v = Float64.(collect(values))
        !isempty(v) && all(isfinite,v) && all(v .> 0) && all(diff(v) .> 0) ||
            throw(ArgumentError("Productivity values must be representable, positive, and strictly increasing"))
        check_transition(transition)
        size(transition,1) == length(v) || throw(DimensionMismatch("Transition size differs from productivity support"))
        new(v,Matrix{Float64}(transition))
    end
end
Base.getproperty(p::MarkovChain,key::Symbol) = _readonly(getfield(p,key))

"""
    LogAR1(; persistence=0.9, innovation_std=0.1, log_mean=0.0)

Individual log productivity follows `x′ = log_mean + persistence*(x-log_mean) + ε`,
with Gaussian innovations of standard deviation `innovation_std`, and `z=exp(x)`.
Require `abs(persistence)<1`. `StateGrid(...; z=(n=7,))` uses the Rouwenhorst
approximation. No normalization of mean productivity is applied: `log_mean=0`
is not a unit mean in levels. Zero volatility collapses to one state `exp(log_mean)`.
"""
struct LogAR1 <: DiscreteProductivity
    persistence::Float64
    innovation_std::Float64
    log_mean::Float64
    function LogAR1(;persistence=0.9,innovation_std=0.1,log_mean=0.0)
        xs=(persistence,innovation_std,log_mean)
        all(x -> x isa Real && !(x isa ForwardDiff.Dual) && isfinite(x),xs) ||
            throw(ArgumentError("AR(1) parameters must be finite real numbers; AD is unsupported"))
        ρ,σ,μ=Float64.(xs)
        all(isfinite,(ρ,σ,μ)) && abs(ρ)<1 && σ>=0 ||
            throw(ArgumentError("Require finite parameters, abs(persistence)<1, and innovation_std>=0"))
        new(ρ,σ,μ)
    end
end

# Rouwenhorst recursion; see Kopecky and Suen (2010), doi:10.1016/j.red.2010.02.002.
function rouwenhorst(p::LogAR1,n)
    n isa Integer && n >= 1 || throw(ArgumentError("Productivity n must be a positive integer"))
    if p.innovation_std == 0
        return MarkovChain(values=[exp(p.log_mean)],transition=ones(1,1))
    end
    n >= 2 || throw(ArgumentError("Positive AR(1) volatility requires at least two states"))
    q=(1+p.persistence)/2
    P=[q 1-q; 1-q q]
    for m in 3:n
        next=zeros(m,m)
        next[1:m-1,1:m-1] .+= q.*P
        next[1:m-1,2:m] .+= (1-q).*P
        next[2:m,1:m-1] .+= (1-q).*P
        next[2:m,2:m] .+= q.*P
        next[2:m-1,:] ./= 2
        P=next
    end
    width=p.innovation_std/sqrt(1-p.persistence^2)*sqrt(n-1)
    return MarkovChain(values=exp.(range(p.log_mean-width,p.log_mean+width;length=n)),transition=P)
end

default_grid_spec(::MarkovChain) = NamedTuple()
default_grid_spec(::LogAR1) = (n=7,)
function process_axis(p::MarkovChain,spec)
    if spec isa NamedTuple
        validate_options(spec,(:n,),"Markov productivity")
        get(spec,:n,length(p.values)) == length(p.values) ||
            throw(DimensionMismatch("Markov support belongs to the process"))
    else
        collect(spec) == p.values || throw(ArgumentError("Markov coordinates must equal process support"))
    end
    return (coordinates=copy(p.values),kind=:categorical)
end
function process_axis(p::LogAR1,spec)
    if spec isa NamedTuple
        validate_options(spec,(:n,),"AR(1) productivity")
        chain=rouwenhorst(p,get(spec,:n,7))
    else
        chain=rouwenhorst(p,length(spec))
        collect(spec) == chain.values || throw(ArgumentError("AR(1) coordinates must match its Rouwenhorst support"))
    end
    return (coordinates=copy(chain.values),kind=:categorical)
end

"""Per-period row-stochastic productivity matrix on the supplied categorical support."""
function transition(p::MarkovChain,coordinates=p.values)
    process_axis(p,coordinates)
    return copy(p.transition)
end
function transition(p::LogAR1,coordinates)
    process_axis(p,coordinates)
    return copy(rouwenhorst(p,length(coordinates)).transition)
end
