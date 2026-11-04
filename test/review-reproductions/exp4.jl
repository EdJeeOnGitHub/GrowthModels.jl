using GrowthModels, Printf
const G=GrowthModels
println("closed form γ=α=.3, Δ=1")
for N in (250,500,1000,2000)
  α=.3;m=RamseyCassKoopmansModel(γ=α,α=α,ρ=.05,δ=.05,A=.6);kss=G.k_steady_state(m);hp=StateSpaceHyperParams((k=HyperParams(N=N,xmin=.3kss,xmax=1.5kss),))
  r=solve_HJB(m,hp;verbose=false,crit=1e-9,Delta=1,maxit=5000);φ=(m.ρ+m.δ*(1-α))/α;k=r.variables.k;c=r.variables.c
  @printf(" N=%d conv=%s sup|c-φk|=%.3e sup interior(excl 3 each end)=%.3e\n",N,r.value.convergence_status,maximum(abs.(c.-φ.*k)),maximum(abs.(c[4:end-3].-φ.*k[4:end-3])))
end
m=SkibaModel();hp=StateSpaceHyperParams(m;Nk=4000);r=solve_HJB(m,hp;verbose=false,crit=1e-9)
k=r.variables.k;lo,hi=G.k_steady_state(m);i=findall(x->lo+.01<x<hi-.01,k);j=findfirst(r.variables.If[i]);println("Skiba: crossing k_star=",round(G.k_star(m),digits=4)," policy threshold (first forward between steady states)=",round(k[i[j]],digits=4), " analytic kss=",round.((lo,hi),digits=4))
