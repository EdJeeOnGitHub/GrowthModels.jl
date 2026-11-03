
struct ValueFunctionError <: Exception
    var::String
end
Base.showerror(io::IO, e::ValueFunctionError) = print(io, "ValueFunctionError: Error whilst iterating value function - $(e.var)")



function generate_dx(x::AbstractVector)
    length(x) >= 2 || throw(ArgumentError("At least two grid points are required"))
    spacing = diff(x)
    return [spacing; spacing[end]], [spacing[1]; spacing]
end

# Choose exactly one admissible drift. At nonconcave points both directions
# can be feasible; the larger Hamiltonian wins (forward on an exact tie).
# The first array dimension is capital, with state constraints at its endpoints.
@inline function upwind_choice(vf, vb, c0, γ, k, Nk)
    cf = max(vf, eps())^(-1 / γ)
    cb = max(vb, eps())^(-1 / γ)
    sf, sb = c0 - cf, c0 - cb
    forward, backward = k < Nk && sf > 0, k > 1 && sb < 0
    Hf = cf^(1 - γ) / (1 - γ) + vf * sf
    Hb = cb^(1 - γ) / (1 - γ) + vb * sb
    If = forward && (!backward || Hf >= Hb)
    Ib = backward && (!forward || Hb > Hf)
    I0 = !(If || Ib)
    return If ? cf : Ib ? cb : c0, If, Ib, I0, sf, sb
end

function select_upwind_policy(dVf, dVb, c0, γ)
    T = promote_type(eltype(dVf), eltype(dVb), eltype(c0), typeof(γ))
    c, sf, sb = similar(c0, T), similar(c0, T), similar(c0, T)
    If, Ib, I0 = falses(size(c0)), falses(size(c0)), falses(size(c0))
    Nk = size(c0, 1)
    for i in eachindex(c0)
        c[i], If[i], Ib[i], I0[i], sf[i], sb[i] =
            upwind_choice(dVf[i], dVb[i], c0[i], γ, mod1(i, Nk), Nk)
    end
    return (; c, If, Ib, I0, sf, sb)
end

# A backward generator must conserve constants and have nonnegative jump rates.
function check_generator(A::SparseMatrixCSC; atol = 1e-6)
    size(A, 1) == size(A, 2) || throw(ValueFunctionError("Generator must be square"))
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in axes(A, 2), ptr in nzrange(A, j)
        a = vals[ptr]
        if !isfinite(a) || (rows[ptr] != j && a < 0)
            throw(ValueFunctionError("Generator has a nonfinite entry or negative off-diagonal rate"))
        end
    end
    maximum(abs, sum(A, dims = 2)) <= atol ||
        throw(ValueFunctionError("Generator rows do not sum to zero"))
    return nothing
end


utility(c, γ) = c^(1 - γ) / (1 - γ)
function validate_utility(m)
    isfinite(preferences(m).γ) && preferences(m).γ > 0 || throw(ArgumentError("Risk aversion γ must be finite and positive"))
    preferences(m).γ != 1 || throw(ArgumentError("γ=1 (log utility) is not supported; the CRRA value normalization is singular there"))
end
initial_guess(m::Model, resources::AbstractArray) = utility.(resources, preferences(m).γ) ./ preferences(m).ρ
