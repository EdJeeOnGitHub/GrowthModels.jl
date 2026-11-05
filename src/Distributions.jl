const PROBABILITY_TOLERANCE = 1e-10

"""A probability distribution on a StateGrid. Use `distribution` to validate input,
and `mass`/`density` to obtain shaped copies. This object carries no time.
"""
struct StateDistribution{G,M}
    grid::G
    probabilities::M
    function StateDistribution(grid::StateGrid, probabilities::AbstractArray)
        size(probabilities) == size(grid) || throw(DimensionMismatch("Probability array must have shape $(size(grid))"))
        validate_probability(probabilities)
        owned = copy(probabilities)
        new{typeof(grid),typeof(owned)}(deepcopy(grid), owned)
    end
end
Base.getproperty(d::StateDistribution,key::Symbol) = _readonly(getfield(d,key))
function validate_probability(p)
    total = sum(p)
    all(x -> isfinite(x) && primal(x) >= 0, p) || throw(ArgumentError("Probabilities must be finite and nonnegative; total mass=$total, tolerance=$PROBABILITY_TOLERANCE"))
    abs(primal(total)-1) <= PROBABILITY_TOLERANCE || throw(ArgumentError("Expected unit mass; measured mass=$total, tolerance=$PROBABILITY_TOLERANCE. Normalize explicitly before construction."))
    return total
end
# Direct solves meet nonnegativity only up to roundoff, which matters for tail
# probabilities near machine precision. Remove that roundoff; a materially
# negative entry means the solve failed and must not be hidden by clipping.
# Mass without negative entries is returned exactly as solved.
function clip_probability_roundoff!(p)
    lowest=minimum(primal,p)
    lowest >= -PROBABILITY_TOLERANCE ||
        throw(ArgumentError("Stationary solve returned a negative probability of $lowest; tolerance=$PROBABILITY_TOLERANCE"))
    lowest < 0 || return p
    for i in eachindex(p)
        primal(p[i]) < 0 && (p[i]=zero(p[i]))
    end
    p ./= sum(p)
    return p
end
function distribution(g::StateGrid; masses=nothing, density=nothing)
    (masses === nothing) != (density === nothing) || throw(ArgumentError("Supply exactly one of masses or density"))
    data = masses === nothing ? density : masses
    data isa AbstractArray && size(data) == size(g) || throw(DimensionMismatch("Input must be a shaped array of size $(size(g))"))
    probabilities = masses === nothing ? data .* getfield(g,:weights) : copy(data)
    return StateDistribution(g, probabilities)
end
mass(d::StateDistribution) = copy(getfield(d,:probabilities))
density(d::StateDistribution) = getfield(d,:probabilities) ./ getfield(d.grid,:weights)
function expectation(d::StateDistribution, f)
    return sum(f(grid_state(d.grid,i))*getfield(d,:probabilities)[i] for i in CartesianIndices(size(d.grid)))
end
function boundary_mass(d::StateDistribution)
    p = getfield(d,:probabilities)
    return (lower=sum(selectdim(p,1,1)), upper=sum(selectdim(p,1,size(p,1))))
end
