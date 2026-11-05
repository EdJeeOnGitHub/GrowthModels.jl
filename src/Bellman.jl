# All Bellman arithmetic is Float64. Singular utility endpoints are evaluated
# explicitly; no consumption floor or continuation extrapolation is introduced.
function discrete_utility(c, γ)
    c < 0 && return -Inf
    c == 0 && return γ < 1 ? 0.0 : -Inf
    return γ == 1 ? log(c) : c^(1-γ)/(1-γ)
end

mutable struct BellmanWorkspace
    budget::Vector{Float64}
    slopes::Vector{Float64}
    next_value::Vector{Float64}
    capital::Vector{Float64}
    consumption::Vector{Float64}
    alternative_capital::Vector{Float64}
    choice_gap::Vector{Float64}
end
function BellmanWorkspace(m, k)
    budget = Float64[discrete_budget(m,x) for x in k]
    all(isfinite,budget) || throw(ArgumentError("Nonfinite available goods"))
    n = length(k)
    BellmanWorkspace(budget, zeros(n-1), zeros(n), zeros(n), zeros(n),
        fill(NaN,n), fill(Inf,n))
end

# Detect distinct local maxima, not neighboring candidates along the same
# increasing branch. At a knot both one-sided derivatives must point inward.
function bellman_local_maximum(x, B, k, slopes, β, γ)
    marginal = B == x ? Inf : (B-x)^(-γ)
    j = clamp(searchsortedlast(k,x),1,length(k)-1)
    left = x == k[j] && j > 1 ? slopes[j-1] : slopes[j]
    right = slopes[j]
    return (x == first(k) || β*left >= marginal) &&
        (x == min(B,last(k)) || β*right <= marginal)
end

function bellman_choice(B, k, v, slopes, β, γ; alternatives=false)
    B >= first(k) || throw(ArgumentError("No feasible capital choice within numerical bounds; lower kmin or change resources"))
    best, action = -Inf, first(k)
    # Candidate local maxima are retained only when diagnostics are requested.
    peaks = alternatives ? Tuple{Float64,Float64}[] : nothing
    for j in eachindex(slopes)
        lo, hi = k[j], min(k[j+1],B)
        lo > hi && break
        slope = slopes[j]
        x = slope > 0 ? clamp(B-(β*slope)^(-1/γ),lo,hi) : lo
        # Endpoint evaluation also protects against roundoff in a stationary
        # candidate, and handles zero-width feasible intervals.
        for kp in (lo,x,hi)
            weight = (kp-k[j])/(k[j+1]-k[j])
            continuation = (1-weight)*v[j] + weight*v[j+1]
            q = discrete_utility(B-kp,γ) + β*continuation
            if q > best || (q == best && kp < action)
                best, action = q, kp
            end
            if alternatives && isfinite(q) &&
                    (kp == x && lo < x < hi || bellman_local_maximum(kp,B,k,slopes,β,γ))
                push!(peaks,(kp,q))
            end
        end
    end
    isfinite(best) || throw(ArgumentError("No finite feasible value: check consumption domain and capital bounds (zero income at zero capital is singular for log utility and risk_aversion >= 1)"))
    alt, runnerup = NaN, -Inf
    if alternatives
        separation = 32eps(max(1.0,abs(action)))
        for (kp,q) in peaks
            if abs(kp-action) > separation && (q > runnerup || (q == runnerup && kp < alt))
                alt, runnerup = kp, q
            end
        end
    end
    return (value=best, capital=action, consumption=B-action,
        alternative_capital=alt, choice_gap=best-runnerup)
end

function bellman_step!(ws, k, v, β, γ; alternatives=false)
    ws.slopes .= diff(v) ./ diff(k)
    all(isfinite,ws.slopes) || throw(ArgumentError("Nonfinite continuation slopes at Float64 precision"))
    for i in eachindex(k)
        q = bellman_choice(ws.budget[i],k,v,ws.slopes,β,γ; alternatives)
        ws.next_value[i] = q.value
        ws.capital[i] = q.capital
        ws.consumption[i] = q.consumption
        ws.alternative_capital[i] = q.alternative_capital
        ws.choice_gap[i] = q.choice_gap
    end
    return maximum(abs, ws.next_value .- v)
end

function policy_transition(k, capital)
    n = length(k)
    rows, cols, weights = Int[], Int[], Float64[]
    for i in eachindex(capital)
        j = clamp(searchsortedlast(k,capital[i]),1,n-1)
        w = (capital[i]-k[j])/(k[j+1]-k[j])
        push!(rows,i,i); push!(cols,j,j+1); push!(weights,1-w,w)
    end
    return sparse(rows,cols,weights,n,n)
end
