# Separable production f(k, z, ability) = z*ability*F(k) for an arbitrary increasing F, such as
# an estimated curve. The credit market's static capital choice,
#     max_{0 < k ≤ λa} s*F(k) - c*k,   s = z*ability, c = r + δ,
# depends on (s, c) only through q = s/c, and on the limit L = λa. A table of the best
# capital node for every (q, L) node pair, built once, makes each choice a lookup. F may
# be non-concave, which is the point of an estimated technology, so the table records the
# global maximum over nodes up to L rather than a first-order condition.

"""
    ScaledProduction(F; kmin, kmax, n=1500, qmin=1e-4, qmax=1e8, nq=1200)

Production `z*ability*F(k)` with `F` increasing and positive on `[kmin, kmax]`. Use it as the
production function of a [`FunctionalModel`](@ref) so that [`CreditModel`](@ref) can choose
capital under a collateral limit. Capital choices are restricted to `n` log-spaced nodes on
`[kmin, kmax]` plus the collateral limit itself; `q = scale/(r+δ)` is tabulated on `nq`
log-spaced points in `[qmin, qmax]` and clamped outside it. Choose `kmax` above the largest
capital any household can hold, `λ` times the top of the wealth grid.
"""
struct ScaledProduction{G}
    F::G
    k::Vector{Float64}
    Fk::Vector{Float64}
    logq::Vector{Float64}
    best::Matrix{Int32}          # best[iq, l]: argmax over nodes j ≤ l of F_j - k_j/q
end

function ScaledProduction(F; kmin, kmax, n=1500, qmin=1e-4, qmax=1e8, nq=1200)
    0 < kmin < kmax && isfinite(kmax) || throw(ArgumentError("Need 0 < kmin < kmax < Inf"))
    0 < qmin < qmax && isfinite(qmax) || throw(ArgumentError("Need 0 < qmin < qmax < Inf"))
    n >= 2 && nq >= 2 || throw(ArgumentError("Need at least two capital and two q nodes"))
    k = exp.(range(log(kmin),log(kmax);length=n))
    Fk = Float64[F(x) for x in k]
    all(x -> isfinite(x) && x > 0,Fk) || throw(ArgumentError("F must be finite and positive on [kmin, kmax]"))
    all(>=(0),diff(Fk)) || throw(ArgumentError("F must be nondecreasing on [kmin, kmax]"))
    logq = collect(range(log(qmin),log(qmax);length=nq))
    best = Matrix{Int32}(undef,nq,n)
    Threads.@threads for iq in 1:nq
        invq = exp(-logq[iq])
        bestj = 1; bestv = Fk[1]-k[1]*invq
        for j in 1:n
            v = Fk[j]-k[j]*invq
            v > bestv && (bestv = v; bestj = j)
            best[iq,j] = bestj
        end
    end
    return ScaledProduction(deepcopy(F),k,Fk,logq,best)
end
(p::ScaledProduction)(k,z,ability) = z*ability*p.F(k)

# F at the collateral limit, by log-log interpolation of the table on its range (the
# table is fine enough that this matches F to the precision of the nodes), and F itself
# outside it.
function table_F(p::ScaledProduction,x)
    k = p.k
    (x <= first(k) || x >= last(k)) && return p.F(x)
    j = searchsortedlast(k,x)
    t = (log(x)-log(k[j]))/(log(k[j+1])-log(k[j]))
    return exp((1-t)*log(p.Fk[j])+t*log(p.Fk[j+1]))
end

# Best capital in (0, L] for profit s*F(k) - c*k: the tabulated node for the q nodes
# bracketing s/c, or the limit L itself, whichever earns more.
function scaled_capital_choice(p::ScaledProduction,s,c,L)
    k,Fk = p.k,p.Fk
    profit(x,Fx) = s*Fx-c*x
    bestk = L; bestF = table_F(p,L); bestv = profit(bestk,bestF)
    l = searchsortedlast(k,L)
    l == 0 && return (capital=bestk,profit=bestv)
    lq = log(s/c)
    iq = clamp(searchsortedlast(p.logq,lq),1,length(p.logq)-1)
    bestj = 0
    for row in (iq,iq+1)
        j = p.best[row,l]
        v = profit(k[j],Fk[j])
        v > bestv && (bestv = v; bestk = k[j]; bestF = Fk[j]; bestj = j)
    end
    # The table is exact only at its q nodes: climb along capital nodes at the actual (s, c),
    # staying at or below the limit, to the local maximum near the tabulated choice.
    if bestj > 0
        while bestj < l && profit(k[bestj+1],Fk[bestj+1]) > bestv
            bestj += 1; bestv = profit(k[bestj],Fk[bestj])
        end
        while bestj > 1 && profit(k[bestj-1],Fk[bestj-1]) > bestv
            bestj -= 1; bestv = profit(k[bestj],Fk[bestj])
        end
        bestk = k[bestj]
        lv = profit(L,table_F(p,L))
        lv > bestv && (bestv = lv; bestk = L; bestj = 0)
    end
    # Off the nodes: the vertex of the parabola through the best node and its neighbours,
    # kept below the limit, if it earns more.
    if 1 < bestj < l
        x0,x1,x2 = k[bestj-1],k[bestj],k[bestj+1]
        y0,y1,y2 = profit(x0,Fk[bestj-1]),bestv,profit(x2,Fk[bestj+1])
        d = (x1-x0)*(y1-y2)-(x1-x2)*(y1-y0)
        if d != 0
            x = x1-((x1-x0)^2*(y1-y2)-(x1-x2)^2*(y1-y0))/(2d)
            if x0 < x < min(x2,L)
                v = profit(x,table_F(p,x))
                v > bestv && (bestv = v; bestk = x)
            end
        end
    end
    return (capital=bestk,profit=bestv)
end
