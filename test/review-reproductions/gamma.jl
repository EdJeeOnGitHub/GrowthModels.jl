using GrowthModels, LinearAlgebra, SparseArrays, Printf
const G=GrowthModels
for γ in (0.3,0.5,0.7,0.9,0.95)
  m=RamseyCassKoopmansModel(γ=γ,α=0.3,ρ=0.05,δ=0.05,A=0.6)
  for (lab,hp) in (("default",StateSpaceHyperParams(m;Nk=500)),)
    for Δ in (1000,100,10,1)
      st=StateSpace(m,hp); v0=Value(st); v0.v[:]=G.initial_guess(m,st)
      r=try solve_HJB(m,hp,st;verbose=false,maxit=3000,init_value=v0,Delta=Δ) catch e; (;err=e) end
      if hasproperty(r,:err); println("γ=$γ Δ=$Δ threw"); continue; end
      c=r.variables===nothing ? [NaN] : r.variables.c
      @printf("γ=%.2f Δ=%-5d conv=%-5s it=%4d maxc=%.3e  nonmono=%d\n",γ,Δ,r.value.convergence_status,r.iter,maximum(c),count(diff(r.value.v).<=0))
    end
  end
end
# trace first breakdown
m=RamseyCassKoopmansModel(γ=0.5); hp=StateSpaceHyperParams(m;Nk=500); st=StateSpace(m,hp); val=Value(st); val.v[:]=G.initial_guess(m,st)
println("trace γ=.5: guess range v[1],v[end]=",val.v[1]," ",val.v[end])
for n in 1:6
  global val
  out=update_v(m,val,st,hp;iter=n,verbose=false,crit=1e-300)
  val=out[1]; k=st[:k]; 
  bad=findall(diff(val.v).<=0)
  println(" iter $n nonmono cells=",length(bad)," first=",isempty(bad) ? 0 : first(bad)," maxdV=",round(maximum(val.dVb),digits=3)," dist=",val.dist[end])
end
