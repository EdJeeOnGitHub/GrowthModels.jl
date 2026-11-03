# Internal construction capability: exported result types have no unchecked
# ordinary positional constructor. Deliberate getfield/internal access is not
# part of the supported public interface.
struct OwnedSnapshot end
const OWNED = OwnedSnapshot()

"""An array inspection view. Use `copy` to obtain editable storage."""
struct ReadOnlyArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end
_readonly(a::AbstractArray) = ReadOnlyArray(a)
_readonly(a::ReadOnlyArray) = a
_readonly(x::NamedTuple) = map(_readonly,x)
_readonly(x::Tuple) = map(_readonly,x)
_readonly(x) = x
_storage(a::ReadOnlyArray) = getfield(a,:storage)
_storage(a::AbstractArray) = a
Base.size(a::ReadOnlyArray) = size(_storage(a))
Base.axes(a::ReadOnlyArray) = axes(_storage(a))
Base.IndexStyle(::Type{ReadOnlyArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.getindex(a::ReadOnlyArray, i...) = getindex(_storage(a),i...)
Base.setindex!(::ReadOnlyArray, args...) = throw(ArgumentError("Read-only array; use copy(array) for editable storage"))
Base.getproperty(::ReadOnlyArray, name::Symbol) = throw(ArgumentError("Read-only array has no public property $name"))
Base.propertynames(::ReadOnlyArray, private::Bool=false) = ()
Base.parent(a::ReadOnlyArray) = a
Base.copy(a::ReadOnlyArray) = copy(_storage(a))
Base.similar(a::ReadOnlyArray, ::Type{T}, dims::Dims) where T = similar(_storage(a),T,dims)
Base.view(a::ReadOnlyArray, i...) = _readonly(view(_storage(a),i...))
Base.reshape(a::ReadOnlyArray, dims::Int...) = _readonly(reshape(_storage(a),dims...))
Base.reshape(a::ReadOnlyArray, dims::Tuple{Vararg{Int}}) = _readonly(reshape(_storage(a),dims))
Base.vec(a::ReadOnlyArray) = _readonly(vec(_storage(a)))
Base.transpose(a::ReadOnlyArray) = _readonly(transpose(_storage(a)))
Base.adjoint(a::ReadOnlyArray) = _readonly(adjoint(_storage(a)))
SparseArrays.sparse(a::ReadOnlyArray{T,2}) where T = copy(sparse(_storage(a)))
# Preserve structured/sparse multiplication without exposing the backing matrix.
Base.:*(a::ReadOnlyArray{T,2}, b::AbstractVector) where T = _storage(a) * _storage(b)
Base.:*(a::ReadOnlyArray{T,2}, b::AbstractMatrix) where T = _storage(a) * _storage(b)
# In particular, reductions of a sparse generator must not scan its dense
# logical shape merely because inspection wraps its storage.
Base.sum(a::ReadOnlyArray;kwargs...) = sum(_storage(a);kwargs...)
Base.sum(f,a::ReadOnlyArray;kwargs...) = sum(f,_storage(a);kwargs...)
Base.maximum(a::ReadOnlyArray;kwargs...) = maximum(_storage(a);kwargs...)
Base.maximum(f,a::ReadOnlyArray;kwargs...) = maximum(f,_storage(a);kwargs...)
Base.minimum(a::ReadOnlyArray;kwargs...) = minimum(_storage(a);kwargs...)
Base.minimum(f,a::ReadOnlyArray;kwargs...) = minimum(f,_storage(a);kwargs...)
