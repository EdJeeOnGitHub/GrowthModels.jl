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
println("period,n,threshold,threshold_error,max_flow_consumption_error,threshold_jump,other_jumps_k")
for (Δ,n) in ((1.0,2000),(0.5,3000),(0.25,4000),(0.1,5000))
    m=DiscreteSkibaModel(discount=exp(-ρ*Δ),depreciation=δ*Δ,risk_aversion=hjb.γ,
        A_H=hjb.A_H*Δ,A_L=hjb.A_L*Δ,α=hjb.α,κ=hjb.κ)
    fit=solve(m,StateGrid(m;k=(n=n,bounds=(0.01,8.0)));maxiters=5000)
    require_converged(fit)
    jumps=policy_jumps(fit)
    escape=only(filter(j->j.reverses_direction,jumps))
    threshold=escape.k
    others=join((round(j.k,digits=3) for j in jumps if !j.reverses_direction)," ")
    gap=maximum(abs(consumption(fit,(k=x,))/Δ-consumption(hjbfit,(k=x,))) for x in samples)
    println(join((Δ,n,threshold,threshold-target,gap,escape.jump,others),","))
    flush(stdout)
end
