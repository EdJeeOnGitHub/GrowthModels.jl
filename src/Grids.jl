"""Realized grid with canonical model-defined axes and capital quadrature weights.
Axis properties return copies; use `coordinates(grid)` for shaped arrays.
"""
struct StateGrid{A,K,S,W}
    axes::A
    kinds::K
    specification::S
    weights::W
end
Base.size(g::StateGrid) = map(length, values(getfield(g, :axes)))
Base.length(g::StateGrid) = prod(size(g))
Base.propertynames(g::StateGrid) = (keys(getfield(g, :axes))..., :kinds, :specification)
function Base.getproperty(g::StateGrid, key::Symbol)
    ax = getfield(g, :axes)
    haskey(ax, key) && return copy(getproperty(ax, key))
    key in (:kinds, :specification) && return deepcopy(getfield(g, key))
    throw(ArgumentError("Grid has no public property $key; use coordinates or cell_weights"))
end
primal(x::Real) = x
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))

function capital_axis(spec::NamedTuple)
    n = spec.n
    n isa Integer && n >= 2 || throw(ArgumentError("Capital grid needs at least two points"))
    lo, hi = spec.bounds
    isfinite(lo) && isfinite(hi) && 0 < lo < hi || throw(ArgumentError("Capital bounds must be finite, positive, and increasing"))
    spacing = get(spec, :spacing, :linear)
    spacing in (:linear, :stretched) || throw(ArgumentError("spacing must be :linear or :stretched"))
    power = spacing == :linear ? 1 : get(spec, :power, 2.0)
    isfinite(power) && power > 0 || throw(ArgumentError("Grid power must be positive"))
    return [lo + (hi-lo) * (t == 0 ? zero(power) : t^power) for t in range(0, 1; length=n)]
end
function StateGrid(m::Model; kwargs...)
    schema = state_schema(m)
    schema in ((:k,), (:k,:z), (:k,:z,:ability)) || throw(ArgumentError("Supported schemas are (k,), (k,z), and (k,z,ability)"))
    all(k -> k in (:k, :z) && k in schema, keys(kwargs)) || throw(ArgumentError("Only model-defined k/z discretizations may be supplied; ability support belongs to the model"))
    defaults = default_grid_spec(m)
    specifications = NamedTuple{keys(defaults)}(map(keys(defaults)) do key
        supplied = get(kwargs, key, NamedTuple())
        supplied isa NamedTuple ? merge(defaults[key], supplied) : copy(supplied)
    end)
    kspec = specifications.k
    k = kspec isa NamedTuple ? capital_axis(kspec) : collect(kspec)
    axes = (k=k,)
    kinds = (k=:continuous,)
    if :z in schema
        p = m.stochasticprocess
        zspec = specifications.z
        z = if p isa PoissonProcess
            zspec isa AbstractVector && collect(zspec) != p.z && throw(ArgumentError("Poisson support belongs to the process"))
            if haskey(kwargs, :z) && kwargs[:z] isa NamedTuple && haskey(kwargs[:z], :n)
                kwargs[:z].n == length(p.z) || throw(DimensionMismatch("Poisson grid size must match the process"))
            end
            collect(p.z)
        elseif p isa OrnsteinUhlenbeckProcess
            if zspec isa AbstractVector
                collect(zspec)
            else
                bounds = get(zspec, :bounds, productivity_bounds(p, zspec.width))
                collect(range(bounds[1], bounds[2]; length=zspec.n))
            end
        else
            zspec isa AbstractVector ? collect(zspec) :
                haskey(zspec, :bounds) ? collect(range(zspec.bounds...; length=zspec.n)) :
                throw(ArgumentError("Custom processes require explicit z coordinates or bounds"))
        end
        axes = merge(axes, (z=z,))
        kinds = merge(kinds, (z=:markov,))
    end
    if :ability in schema
        axes = merge(axes, (ability=copy(population(m).values),))
        kinds = merge(kinds, (ability=:markov,))
    end
    for (name, axis) in pairs(axes)
        !isempty(axis) && all(isfinite, axis) && all(diff(axis) .> 0) || throw(ArgumentError("Axis $name must be finite and strictly increasing"))
        name == :k && (length(axis) < 2 || first(axis) <= 0) && throw(ArgumentError("Capital requires two or more positive points"))
    end
    if :z in schema && m.stochasticprocess isa OrnsteinUhlenbeckProcess
        length(axes.z) >= 2 && first(axes.z) > 0 || throw(ArgumentError("OU productivity requires at least two positive points"))
    end
    dims = map(length, values(axes))
    weights = reshape(create_dx_stacked(axes.k, prod(dims) ÷ length(axes.k)), dims)
    return StateGrid(deepcopy(axes), kinds, deepcopy(specifications), weights)
end
cell_weights(g::StateGrid) = copy(getfield(g, :weights))
function coordinates(g::StateGrid)
    ax = getfield(g, :axes)
    dims = size(g)
    return NamedTuple{keys(ax)}(ntuple(length(dims)) do j
        repeat(reshape(ax[j], ntuple(d -> d == j ? dims[j] : 1, length(dims))),
            ntuple(d -> d == j ? 1 : dims[d], length(dims))...)
    end)
end
function grid_state(g::StateGrid, i::CartesianIndex)
    ax = getfield(g, :axes)
    return NamedTuple{keys(ax)}(ntuple(d -> ax[d][i[d]], length(i.I)))
end
function compatible_grid(a::StateGrid, b::StateGrid)
    aa, bb = getfield(a, :axes), getfield(b, :axes)
    return keys(aa) == keys(bb) && a.kinds == b.kinds &&
        all(i -> length(aa[i]) == length(bb[i]) && all(primal.(aa[i]) .== primal.(bb[i])), eachindex(aa))
end
function validate_model_grid(m, g)
    keys(getfield(g, :axes)) == state_schema(m) || throw(ArgumentError("Grid schema differs from model"))
    if population(m) !== nothing
        primal.(population(m).values) == primal.(g.ability) || throw(ArgumentError("Grid ability support differs from model population"))
    end
    if !is_deterministic(m) && m.stochasticprocess isa PoissonProcess
        primal.(m.stochasticprocess.z) == primal.(g.z) || throw(ArgumentError("Grid productivity support differs from Poisson process"))
    end
end

# Private bridge to the tested finite-difference assembly. No public legacy types.
function numerical_state(m, g)
    ax = getfield(g, :axes)
    legacy = haskey(ax, :ability) ? (k=ax.k, z=ax.z, η=ax.ability) : ax
    y = map(i -> resources(m, grid_state(g, i)), CartesianIndices(size(g)))
    st = StateSpace(legacy, (y=y,))
    hp = StateSpaceHyperParams(map(a -> HyperParams(N=length(a), xmin=first(a), xmax=last(a)), st.state))
    diffusion = if is_deterministic(m)
        spzeros(eltype(y), length(g), length(g))
    else
        Q = sparse(generator(m.stochasticprocess, ax.z))
        size(Q) == (length(ax.z),length(ax.z)) || throw(DimensionMismatch("Process generator must match its axis"))
        check_generator(Q)
        joint = kron(Q, sparse(I,length(ax.k),length(ax.k)))
        haskey(ax,:ability) ? kron(sparse(I,length(ax.ability),length(ax.ability)),joint) : joint
    end
    return st, hp, diffusion
end
