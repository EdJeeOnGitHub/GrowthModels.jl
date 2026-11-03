using GrowthModels, LinearAlgebra, SparseArrays, ForwardDiff, Printf, Statistics
const G = GrowthModels
println("== A gamma<1 spurious c")
let
 for (γ,lab) in ((0.3,"γ=.3"),(0.5,"γ=.5"),(0.9,"γ=.9"),(1.5,"γ=1.5"),(2.0,"γ=2"))
  m=RamseyCassKoopmansModel(γ=γ,α=0.3,ρ=0.05,δ=0.05,A=0.6); kss=G.k_steady_state(m)
  hp=StateSpaceHyperParams((k=HyperParams(N=500,xmin=0.3kss,xmax=1.5kss),))
  r=solve_HJB(m,hp;verbose=false,crit=1e-10); c=r.variables.c; k=r.variables.k
  bad=findall(c.>10*maximum(r.variables.y))
  println(lab," conv=",r.value.convergence_status," bad cells=",bad, isempty(bad) ? "" : " c=$(c[bad]) k=$(k[bad]) vb=$(r.value.dVb[bad]) vf=$(r.value.dVf[bad]) Ib=$(r.variables.Ib[bad]) v[diff]=$(r.value.v[bad].-r.value.v[bad.-1])")
  full=c.^(1-γ)./(1-γ).+r.value.A*r.value.v.-m.ρ.*r.value.v
  @printf("   max residual=%.2e   max|dVb<=0| count=%d  mono v=%s\n",maximum(abs.(full)),count(r.value.dVb[2:end].<=0),all(diff(r.value.v).>0))
  # compare with default-grid & default init
 end
 # default grid with γ<1
 m=SkibaModel(γ=0.5); hp=StateSpaceHyperParams(m;Nk=500); r=solve_HJB(m,hp;verbose=false)
 println(" Skiba γ=.5 default-grid conv=",r.value.convergence_status," maxc=",r.variables===nothing ? nothing : maximum(r.variables.c)," iter=",r.iter)
 m=RamseyCassKoopmansModel(γ=.5); hp=StateSpaceHyperParams(m;Nk=500); r=solve_HJB(m,hp;verbose=false)
 println(" RCK γ=.5 default-grid conv=",r.value.convergence_status," maxc=",r.variables===nothing ? nothing : maximum(r.variables.c)," iter=",r.iter)
 # starting guess sign
 m=RamseyCassKoopmansModel(γ=.3,α=.3); kss=G.k_steady_state(m); hp=StateSpaceHyperParams((k=HyperParams(N=500,xmin=.3kss,xmax=1.5kss),)); st=StateSpace(m,hp)
 gv=G.initial_guess(m,st); println(" init guess first diffs positive? ",all(diff(gv).>0))
 for (lab,iv) in (("guess",nothing),("zeros",Value(st)))
   r=solve_HJB(m,hp,st;verbose=false,crit=1e-10,init_value=iv); println("  $lab maxc=",maximum(r.variables.c)," argmax=",argmax(r.variables.c)," Nk=500")
 end
end
println("== B AD vs FD")
hpd=StateSpaceHyperParams((k=HyperParams(N=200,xmin=.1,xmax=8.),))
hps=StateSpaceHyperParams((k=HyperParams(N=100,xmin=.1,xmax=8.),z=HyperParams(N=2,xmin=.1,xmax=.2)))
hpa=StateSpaceHyperParams((k=HyperParams(N=60,xmin=.1,xmax=8.),z=HyperParams(N=2,xmin=.1,xmax=.2),η=HyperParams(N=2,xmin=.9,xmax=1.1)))
hpo=StateSpaceHyperParams((k=HyperParams(N=60,xmin=.1,xmax=8.),z=HyperParams(N=6,xmin=.8,xmax=1.2)))
Q=[-.2 .2;.3 -.3]
pp(p)=PoissonProcess(z=[0.1*one(p),0.2*one(p)],Q=Q.*one(p))
function run(f,x)
  d=ForwardDiff.derivative(f,x); h=1e-5; (d,(f(x+h)-f(x-h))/2h)
end
cases=[
 ("RCK d/dρ", p->(o=one(p);solve_HJB(RamseyCassKoopmansModel(2o,.3o,p,.05o,.6o),hpd;verbose=false,crit=1e-11).value.v[100]),.05),
 ("Skiba dv/dκ", p->(o=one(p);solve_HJB(SkibaModel(2o,.3o,.05o,.05o,.6o,.4o,p),hpd;verbose=false,crit=1e-11).value.v[100]),2.0),
 ("Skiba dc/dκ", p->(o=one(p);solve_HJB(SkibaModel(2o,.3o,.05o,.05o,.6o,.4o,p),hpd;verbose=false,crit=1e-11).variables.c[100]),2.0),
 ("Stoch Poisson dv/dγ", p->(o=one(p);solve_HJB(StochasticSkibaModel(pp(p);γ=p,α=.3o,ρ=.05o,δ=.05o,A_H=.6o,A_L=.4o,κ=2o),hps;verbose=false,crit=1e-11).value.v[50,2]),2.0),
 ("Stoch OU dv/dσ", p->(o=one(p);solve_HJB(StochasticSkibaModel(OrnsteinUhlenbeckProcess(θ=.1o,σ=p);γ=2o,α=.3o,ρ=.05o,δ=.05o,A_H=.6o,A_L=.4o,κ=2o),hpo;verbose=false,crit=1e-11).value.v[30,3]),.1),
 ("Ability dv/dA_H", p->(o=one(p);solve_HJB(StochasticSkibaAbilityModel(pp(p);γ=2o,α=.3o,ρ=.05o,δ=.05o,A_H=p,A_L=.4o,κ=2o,η_mean=o),hpa;verbose=false,crit=1e-11).value.v[30,2,2]),.6),
 ("flat_transfer dv/dτ", τ->solve_HJB(RamseyCassKoopmansModel(),hpd;verbose=false,crit=1e-11,flat_transfer=τ).value.v[100],0.01),
 ("default-grid Skiba dc/dα", p->(o=one(p);m=SkibaModel(2o,p,.05o,.05o,.6o,.4o,2o);solve_HJB(m,StateSpaceHyperParams(m;Nk=200);verbose=false,crit=1e-11).variables.c[100]),.3),
]
for (n,f,x) in cases
  try; d,fd=run(f,x); @printf(" %-26s AD=%.6f FD=%.6f rel=%.1e\n",n,d,fd,abs(d-fd)/max(abs(fd),1e-12)); catch err; println(" $n threw: ",first(sprint(showerror,err),250)); end
end
println("== C adjoint timing at scale")
let n=30000; d=fill(1.0,n-1); A=spdiagm(-1=>d,1=>d,0=>[-1;fill(-2.0,n-2);-1]); ws=EvolutionWorkspace(sparse(A'),ones(n))
 @printf(" sparse %.4fs\n",@elapsed update_evolution!(ws,sparse(A'))); @printf(" Adjoint %.3fs\n",@elapsed update_evolution!(ws,A'))
end
println("== D default dt dependence on observation times")
let m=StochasticSkibaModel(); hp=StateSpaceHyperParams(m;Nk=300,Nz=10); r=solve_HJB(m,hp;verbose=false); sm=SolvedModel(m,r); N=length(r.value.v)
  k=vec(r.variables.k); w=create_dx_stacked(r.variables.k[:,1],10); g=ones(N)
  mean_at(ev,t)=dot(k,ev.cell_masses[:,findfirst(==(t),ev.times)])
  for ts in ([0.,20.],[0.,5.,10.,20.],collect(0.:1.:20.),collect(0.:.1:20.))
    ev=StateEvolution(g,sm,ts;save=:all); @printf(" %3d obs: E[k(20)]=%.5f\n",length(ts),mean_at(ev,20.0))
  end
end
println("== E stationary with normalize=false")
let A=sparse([-1.0 1;1 -1]);w=[.5,2.];ev=StateEvolution([4.0,0.],A,[0.,1.],(2,),w;normalize=false,stationary=true)
 println(" final mass=",weighted_mass(ev.final_density,w)," stationary mass=",weighted_mass(ev.stationary_dist,w)); end
println("== F custom 3-d diffusion coupling")
let k=[1.,2.,3.];z=[1.,2.];η=[1.,2.];y=ones(3,2,2).+.05 .*reshape(k,3,1,1)
 m=StochasticSkibaAbilityModel(PoissonProcess(z=z,Q=[-.1 .1;.1 -.1]));hp=StateSpaceHyperParams((k=HyperParams(N=3,xmin=1.,xmax=3.),z=HyperParams(N=2,xmin=1.,xmax=2.),η=HyperParams(N=2,xmin=1.,xmax=2.)));st=StateSpace((k=k,z=z,η=η),(y=y,))
 B=kron(sparse([-.1 .1;.1 -.1]),sparse(I,6,6))*1.0
 try; update_v(m,Value(st),st,hp,B;verbose=false); println(" coupled ok"); catch err; println(" coupled ability-switch via update_v: ",first(sprint(showerror,err),120)); end
 try; update_v(m,Value(st),st,hp,B;verbose=false,linear_solver=:sparse); println(" with :sparse ok"); catch err; println(" sparse threw ",first(sprint(showerror,err),120)); end
end
