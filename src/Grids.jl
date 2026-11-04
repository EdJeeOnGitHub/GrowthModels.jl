function create_dx_stacked(x, N_second_dim)
    dxf, dxb = generate_dx(x)
    dx_tilde = 0.5 * (dxf + dxb)
    dx_tilde[1] = 0.5*dxf[1]
    dx_tilde[end] = 0.5*dxb[end]
    dx_tilde_stacked = repeat(dx_tilde, N_second_dim)
    return dx_tilde_stacked
end


"""Realized grid with canonical model-defined axes and capital quadrature weights.
Axis properties return read-only views; use `coordinates(grid)` for shaped arrays.
"""
struct StateGrid{A,K,S,W}
    axes::A
    kinds::K
    specification::S
    weights::W
    function StateGrid(::OwnedSnapshot,axes,kinds,specification,weights)
        new{typeof(axes),typeof(kinds),typeof(specification),typeof(weights)}(axes,kinds,specification,weights)
    end
end
Base.size(g::StateGrid) = map(length, values(getfield(g, :axes)))
Base.length(g::StateGrid) = prod(size(g))
Base.propertynames(g::StateGrid) = (keys(getfield(g, :axes))..., :kinds, :specification)
function Base.getproperty(g::StateGrid, key::Symbol)
    ax = getfield(g, :axes)
    haskey(ax, key) && return _readonly(getproperty(ax, key))
    key in (:kinds, :specification) && return deepcopy(getfield(g, key))
    throw(ArgumentError("Grid has no public property $key; use coordinates or cell_weights"))
end
primal(x::Real) = x
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))

function capital_axis(spec::NamedTuple)
    validate_options(spec, (:n,:bounds,:spacing,:power), "capital axis")
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
    # Fully explicit coordinates/bounds bypass model defaults entirely.
    supplied_k = get(kwargs,:k,NamedTuple())
    supplied_k isa NamedTuple && validate_options(supplied_k,(:n,:bounds,:spacing,:power),"capital axis")
    needs_defaults = supplied_k isa NamedTuple && !(haskey(supplied_k,:n) && haskey(supplied_k,:bounds))
    defaults = needs_defaults ? default_grid_spec(m) : nothing
    kspec = needs_defaults ? merge(defaults.k,supplied_k) : supplied_k
    k = kspec isa NamedTuple ? capital_axis(kspec) : collect(kspec)
    axes = (k=k,)
    kinds = (k=:interpolable,)
    specifications = (k=kspec,)
    if :z in schema
        p = process(m)
        supplied_z = get(kwargs,:z,nothing)
        # A complete explicit z specification must not evaluate unused defaults.
        complete_z = supplied_z !== nothing && (!(supplied_z isa NamedTuple) ||
            (haskey(supplied_z,:bounds) && haskey(supplied_z,:n)))
        zspec = if complete_z
            supplied_z
        else
            proposals = defaults === nothing ? default_grid_spec(m) : defaults
            base = get(proposals,:z,default_grid_spec(p))
            supplied_z === nothing ? base : merge(base,supplied_z)
        end
        resolved = process_axis(p,zspec)
        resolved.kind in (:interpolable,:categorical) || throw(ArgumentError("Unknown process axis kind $(resolved.kind)"))
        axes = merge(axes,(z=collect(resolved.coordinates),))
        kinds = merge(kinds,(z=resolved.kind,))
        specifications = merge(specifications,(z=zspec,))
    end
    if :ability in schema
        axes = merge(axes, (ability=copy(population(m).values),))
        kinds = merge(kinds, (ability=:categorical,))
    end
    if :z in schema && supplied_k isa NamedTuple && !haskey(supplied_k,:bounds)
        bounds=resolved_capital_bounds(m,axes.z,get(axes,:ability,nothing))
        if bounds !== nothing
            kspec=merge(kspec,(bounds=bounds,))
            axes=merge(axes,(k=capital_axis(kspec),))
            specifications=merge(specifications,(k=kspec,))
        end
    end
    for (name, axis) in pairs(axes)
        !isempty(axis) && all(isfinite, axis) && all(diff(axis) .> 0) || throw(ArgumentError("Axis $name must be finite and strictly increasing"))
        name == :k && (length(axis) < 2 || first(axis) <= 0) && throw(ArgumentError("Capital requires two or more positive points"))
    end
    dims = map(length, values(axes))
    weights = reshape(create_dx_stacked(axes.k, prod(dims) ÷ length(axes.k)), dims)
    return StateGrid(OWNED, deepcopy(axes), kinds, deepcopy(specifications), weights)
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
    if process(m) !== nothing
        resolved = process_axis(process(m),getfield(g,:axes).z)
        resolved.kind == getfield(g,:kinds).z || throw(ArgumentError("Grid productivity semantics differ from process"))
    end
end

# Multilinear interpolation on named axes. The scalar arithmetic (including
# moving knots) retains Dual information. Bracket selection is piecewise smooth.
function interpolate_grid(values, g::StateGrid, state::NamedTuple; extrapolate=false)
    ax = getfield(g, :axes)
    Set(keys(state)) == Set(keys(ax)) || throw(ArgumentError("State keys must match $(keys(ax))"))
    brackets = map(keys(ax)) do key
        a, x = ax[key], state[key]
        if getfield(g,:kinds)[key] == :categorical
            j = findfirst(y -> primal(y) == primal(x),a)
            j === nothing && throw(ArgumentError("$key must be a represented categorical state"))
            return (j,j,zero(x-a[j]))
        end
        if length(a) == 1
            primal(x) == primal(a[1]) || throw(ArgumentError("Singleton axis $key only supports its one state"))
            return (1, 1, zero(x-a[1]))
        end
        !extrapolate && !(primal(first(a)) <= primal(x) <= primal(last(a))) &&
            throw(ArgumentError("State outside $key support; extrapolation requires explicit opt-in"))
        j = clamp(searchsortedlast(primal.(a), primal(x)), 1, length(a)-1)
        return (j, j+1, (x-a[j])/(a[j+1]-a[j]))
    end
    result = zero(values[firstindex(values)] * (1 + last(first(brackets))))
    for corner in CartesianIndices(ntuple(_ -> 2, length(brackets)))
        index = ntuple(d -> brackets[d][corner[d]], length(brackets))
        weight = prod(d -> corner[d] == 1 ? 1-brackets[d][3] : brackets[d][3], 1:length(brackets))
        result += weight * values[index...]
    end
    return result
end

function validate_regrid(oldgrid::StateGrid,newgrid::StateGrid)
    old,new=getfield(oldgrid,:axes),getfield(newgrid,:axes)
    keys(old) == keys(new) || throw(ArgumentError("Regridding cannot change the state schema"))
    getfield(oldgrid,:kinds) == getfield(newgrid,:kinds) || throw(ArgumentError("Regridding cannot change axis semantics"))
    for key in keys(old)
        if getfield(oldgrid,:kinds)[key] == :categorical
            primal.(old[key]) == primal.(new[key]) || throw(ArgumentError("Regridding cannot change categorical support"))
        end
    end
end

# Canonical independent ability blocks; actual operator coupling is checked by
# independent_blocksize before using a block solver.
preferred_blocksize(g::StateGrid) = length(size(g)) > 2 ? prod(size(g)[1:2]) : length(g)
