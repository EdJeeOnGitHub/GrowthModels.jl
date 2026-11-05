# Run with: julia --project=. examples/discrete_continuous_limit.jl
# Compares a Skiba economy with period length Δ against the existing HJB solution.
# Budget c + k′ = Δf(k) + (1-δΔ)k and discount exp(-ρΔ). CRRA homogeneity makes the
# policy independent of the constant scale between period consumption and flow
# reward, so c/Δ approaches the HJB policy. The package performs no such conversion.
using GrowthModels

ρ,δ=0.05,0.05
hjb=SkibaModel(ρ=ρ,δ=δ)
hjbgrid=StateGrid(hjb;k=(n=4000,bounds=(0.01,8.0)))
hjbfit=solve(hjb,hjbgrid;maxiters=5000)
require_converged(hjbfit)
drift=hjbfit.policy.drift
target=only([hjbgrid.k[i] for i in 1:length(drift)-1 if hjbgrid.k[i]>2 && drift[i]<0 && drift[i+1]>0])
println("hjb_threshold=",target)
samples=(1.5,2.0,2.5,4.0,5.0)
println("period,n,threshold,threshold_error,max_flow_consumption_error")
for (Δ,n) in ((1.0,2000),(0.5,3000),(0.25,4000),(0.1,5000))
    m=DiscreteSkibaModel(discount=exp(-ρ*Δ),depreciation=δ*Δ,risk_aversion=hjb.γ,
        A_H=hjb.A_H*Δ,A_L=hjb.A_L*Δ,α=hjb.α,κ=hjb.κ)
    fit=solve(m,StateGrid(m;k=(n=n,bounds=(0.01,8.0)));maxiters=5000)
    require_converged(fit)
    # The bracket excludes knot-level sign changes beside the low steady state.
    lo,hi=2.0,2.8
    for _ in 1:45
        mid=(lo+hi)/2
        next_capital(fit,(k=mid,))>mid ? (hi=mid) : (lo=mid)
    end
    threshold=(lo+hi)/2
    gap=maximum(abs(consumption(fit,(k=x,))/Δ-consumption(hjbfit,(k=x,))) for x in samples)
    println(join((Δ,n,threshold,threshold-target,gap),","))
    flush(stdout)
end
