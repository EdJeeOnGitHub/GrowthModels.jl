
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


"""
    update_value_function!(init_value, res)

If the HJB solver has converged, update the value function inplace. Uses ForwardDiff.value 
to ensure only the value and not dual components are passed. 

update_v is internal function used for value function iteration. update_value_function! is exported to the user so we can hotstart the VFI across optimization draws.
"""
function update_value_function!(init_value, res)
    if res.value.convergence_status == true
            init_value.v[:] = ForwardDiff.value.(res.value.v)
            init_value.dVf[:] = ForwardDiff.value.(res.value.dVf)
            init_value.dVb[:] = ForwardDiff.value.(res.value.dVb)
            init_value.dV0[:] = ForwardDiff.value.(res.value.dV0)
    end
end


"""
construct_diffusion_matrix(stochasticprocess::OrnsteinUhlenbeckProcess, state::StateSpace, hyperparams::StateSpaceHyperParams)

Constructs the diffusion matrix for the HJB equation.

# Arguments
- `stochasticprocess::OrnsteinUhlenbeckProcess`: The Ornstein-Uhlenbeck process.
- `state::StateSpace`: The state space.
- `hyperparams::StateSpaceHyperParams`: The hyperparameters of the state space.

# Returns
- `Bswitch`: The diffusion matrix.
"""
function construct_diffusion_matrix(stochasticprocess::OrnsteinUhlenbeckProcess, state::StateSpace, hyperparams::StateSpaceHyperParams)
    (; θ, σ) = stochasticprocess
    z = state[:z]
    k_hps, z_hps = hyperparams[:k], hyperparams[:z]

    state_size = size(state)
    # get size of all other states, but remove z
    Nk, kmax, kmin = k_hps.N, k_hps.xmax, k_hps.xmin
    Nz, zmax, zmin = z_hps.N, z_hps.xmax, z_hps.xmin

    dz_emp = diff(z)
    if (maximum(dz_emp) - minimum(dz_emp)) > 10^(-6)
        throw("Non-uniform grid spacing for z")
    end
    all(isfinite, z) && all(z .> 0) && all(dz_emp .> 0) ||
        throw(ArgumentError("OU productivity grid must be positive and strictly increasing"))
    # for now, assume dz constant (no non-uniform grid for z)
    dz = (zmax - zmin) / (Nz - 1)

    dz2 = dz^2


    σ_sq = σ^2
    # drift
    mu = (-θ*log.(z) .+ σ_sq/2).*z
    # variance - Ito's
    s2 = σ_sq.*z.^2;

    # Upwind the drift in its own direction so all jump rates are nonnegative.
    chi = s2 / (2 * dz2) .+ max.(-mu, 0) ./ dz
    zeta = s2 / (2 * dz2) .+ max.(mu, 0) ./ dz
    yy = .-(chi .+ zeta)
    
    off_diag_length = (Nz-1)*Nk

    ldiag = Vector{typeof(chi[2])}(undef, off_diag_length)
    cdiag = Vector{typeof(yy[1] + chi[1])}(undef, Nz*Nk)
    udiag = Vector{eltype(zeta[1])}(undef, off_diag_length)

    for j in 2:Nz
        ldiag[(j-2)*Nk+1:(j-1)*Nk] .= chi[j]
    end
    for j in 1:(Nz-1)
        udiag[(j-1)*Nk+1:(j)*Nk] .= zeta[j]
    end
    cdiag[1:Nk] .= chi[1] + yy[1]
    for j in 2:(Nz -1)
        cdiag[(j-1)*Nk+1:(j)*Nk] .= yy[j]
    end
    cdiag[end-Nk+1:end] .= zeta[end] + yy[end]

    Bswitch_2dim = spdiagm(-Nk => ldiag, 0 => cdiag, Nk => udiag)
    check_generator(Bswitch_2dim)
    # if only two dimensions (i.e. k and z), then Bswitch is a tridiagonal matrix
    if length(state_size) == 2
        return Bswitch_2dim
    else
        # if more than two dimensions, then Bswitch is a block tridiagonal matrix
        # where there are gaps between the blocks of size Nk
        N_other_states = prod(state_size[3:end])
        Bswitch = kron(sparse(I, N_other_states, N_other_states), Bswitch_2dim)

        return Bswitch
    end
end




"""
    construct_diffusion_matrix(stochasticprocess::PoissonProcess, state::StateSpace, hyperparams::StateSpaceHyperParams)

Constructs the diffusion matrix for the HJB equation.

# Arguments
- `stochasticprocess::PoissonProcess`: The Poisson process representing the stochastic component of the model.
- `state::StateSpace`: The state space of the model.
- `hyperparams::StateSpaceHyperParams`: The hyperparameters of the state space.

# Returns
- `B_switch`: The diffusion matrix.
"""
function construct_diffusion_matrix(stochasticprocess::PoissonProcess, state::StateSpace, hyperparams::StateSpaceHyperParams) 
    (; Q) = stochasticprocess
    state_size = size(state)
    D = size(hyperparams)
    N_k = D[1]
    I_A = sparse(I, N_k, N_k)
    B_switch = kron(Q, I_A)

    if length(state_size) > 2
        N_other_states = prod(state_size[3:end])
        B_switch = kron(sparse(I, N_other_states, N_other_states), B_switch)
    end
    return B_switch
end

utility(c, γ) = c^(1 - γ) / (1 - γ)
function validate_utility(m)
    isfinite(m.γ) && m.γ > 0 || throw(ArgumentError("Risk aversion γ must be finite and positive"))
    m.γ != 1 || throw(ArgumentError("γ=1 (log utility) is not supported; the CRRA value normalization is singular there"))
end
initial_guess(m::GrowthModels.Model, state) = utility.(state.aux_state.y, m.γ) ./ m.ρ

dV_Upwind(::Model, value::Value, variables::NamedTuple) = value.dVf .* variables.If .+ value.dVb .* variables.Ib .+ value.dV0 .* variables.I0
V_err(m::Model) = (value::Value, variables::NamedTuple) -> reshape(
    utility.(vec(variables.c), m.γ) .+ value.A * vec(value.v) .- m.ρ .* vec(value.v), size(value.v))






include("HJBWorkspace.jl")
