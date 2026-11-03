using GrowthModels, LinearAlgebra, SparseArrays, ForwardDiff, Printf, Statistics
const G = GrowthModels
println("== 1 analytic RCK gamma=alpha: c = phi k")
for N in (250,500,1000,2000,4000)
  α=0.3; m=RamseyCassKoopmansModel(γ=α,α=α,ρ=0.05,δ=0.05,A=0.6)
  kss=G.k_steady_state(m); hp=StateSpaceHyperParams((k=HyperParams(N=N,xmin=0.3kss,xmax=1.5kss),))
  r=solve_HJB(m,hp;verbose=false,crit=1e-10); φ=(m.ρ+m.δ*(1-α))/α
  k=r.variables.k; c=r.variables.c
  mid=2:N-1
  @printf("N=%5d conv=%s it=%d  max|c-phi k|=%.3e  med=%.3e\n",N,r.value.convergence_status,r.iter,maximum(abs.(c[mid].-φ.*k[mid])),median(abs.(c[mid].-φ.*k[mid])))
end
println("== 2 steady state vs analytic (grid-level)")
for (name,m) in (("RCK",RamseyCassKoopmansModel()),("Skiba",SkibaModel()))
  hp=StateSpaceHyperParams(m;Nk=2000); r=solve_HJB(m,hp;verbose=false)
  sdot=r.variables.y.-m.δ.*r.variables.k.-r.variables.c
  idx=findall(r.variables.I0)
  println(name," I0 k=",round.(r.variables.k[idx],digits=4)," analytic ",G.k_steady_state(m))
end
println("== 3 init_value zeros vs default (solve_growth_model path)")
for (name,m) in (("RCK",RamseyCassKoopmansModel()),("Skiba",SkibaModel()),("SmoothSkiba",SmoothSkibaModel()),
   ("StochSkibaOU",StochasticSkibaModel()),("StochRCK-OU",StochasticRamseyCassKoopmansModel(OrnsteinUhlenbeckProcess(θ=-log(0.9),σ=0.1))))
  hp= m isa DeterministicModel ? StateSpaceHyperParams(m;Nk=500) : StateSpaceHyperParams(m;Nk=200,Nz=10)
  st=StateSpace(m,hp)
  a=solve_HJB(m,hp,st;verbose=false)
  b=solve_HJB(m,hp,st;verbose=false,init_value=Value(st))
  @printf("%-14s default it=%3d conv=%s | zeros it=%3d conv=%s  max|dv|=%.2e max|dc|=%.2e\n",name,a.iter,a.value.convergence_status,b.iter,b.value.convergence_status,
     maximum(abs.(a.value.v.-b.value.v)), (a.variables===nothing||b.variables===nothing) ? NaN : maximum(abs.(a.variables.c.-b.variables.c)))
end
println("== 4 auto vs sparse")
p=PoissonProcess(z=[0.0,0.1],Q=[-0.2 0.2;0.3 -0.3])
for (name,m,hp) in (("Skiba",SkibaModel(),StateSpaceHyperParams(SkibaModel();Nk=800)),
   ("AbilityOU",StochasticSkibaAbilityModel(),StateSpaceHyperParams(StochasticSkibaAbilityModel();Nk=100,Nz=6,Nη=3)))
  t1=@elapsed a=solve_HJB(m,hp;verbose=false,crit=1e-9)
  t2=@elapsed b=solve_HJB(m,hp;verbose=false,crit=1e-9,linear_solver=:sparse)
  @printf("%-10s dv=%.2e dc=%.2e it %d/%d t %.3f/%.3f\n",name,maximum(abs.(a.value.v-b.value.v)),maximum(abs.(a.variables.c-b.variables.c)),a.iter,b.iter,t1,t2)
end
println("== 5 OU grid truncation")
for (θ,σ) in ((-log(0.9),0.1),)
  pr=OrnsteinUhlenbeckProcess(θ=θ,σ=σ); sdlog=sqrt(σ^2/(2θ))
  m=StochasticSkibaModel(pr); hp=StateSpaceHyperParams(m;Nk=3,Nz=40); st=StateSpace(m,hp)
  z=st[:z]; println(" default z range ",round.((z[1],z[end]),digits=3)," = log ",round.((log(z[1])/sdlog,log(z[end])/sdlog),digits=2)," stationary sds; sd log z=",round(sdlog,digits=3))
  A=G.construct_diffusion_matrix(pr,st,hp)
  Az=A[1:3:end,1:3:end]
  πz=StationaryDistribution(sparse(Az'),ones(40)); πz./=sum(πz)
  mlog=dot(πz,log.(z)); sl=sqrt(dot(πz,(log.(z).-mlog).^2))
  @printf(" discretized: E[z]=%.4f (true %.4f)  sd(log z)=%.4f (true %.4f)\n",dot(πz,z),exp(sdlog^2/2),sl,sdlog)
  zw=range(exp(-4sdlog),exp(4sdlog),length=400); hpw=StateSpaceHyperParams((k=HyperParams(N=2,xmin=1.,xmax=2.),z=HyperParams(N=400,xmin=first(zw),xmax=last(zw))))
  stw=StateSpace((k=[1.,2.],z=collect(zw)),(y=ones(2,400),))
  Aw=G.construct_diffusion_matrix(pr,stw,hpw)[1:2:end,1:2:end]
  πw=StationaryDistribution(sparse(Aw'),ones(400)); πw./=sum(πw)
  mlog=dot(πw,log.(zw)); sl=sqrt(dot(πw,(log.(zw).-mlog).^2))
  @printf(" wide grid 400pts: E[z]=%.4f sd(log z)=%.4f\n",dot(πw,zw),sl)
end
println("== 6 kmax binding by type (default ability)")
m=StochasticSkibaAbilityModel(); hp=StateSpaceHyperParams(m;Nk=300,Nz=10,Nη=3); r=solve_HJB(m,hp;verbose=false)
sm=SolvedModel(m,r); s=sm.variables.y.-m.δ.*sm.variables.k.-sm.variables.c
println(" kmax=",hp[:k].xmax)
pm=s[end-1,:,:] ; println(" share of (z,eta) with saving>0 next to kmax: ",mean(r.variables.If[end-1,:,:]))
println(" If[Nk-1,z,eta]:"); display(Int.(r.variables.If[end-1,:,:]))
Nk,Nz,Nη=size(s); w=create_dx_stacked(sm.variables.k[:,1,1],Nz*Nη)
g=ones(length(w)); sd=StationaryDistribution(sm;initial_density=g).*w; M=reshape(sd,Nk,Nz,Nη)
println(" stationary mass at top capital cell by eta given eta: ",round.(vec(sum(M[end,:,:],dims=1))./vec(sum(M,dims=(1,2))),digits=4))
println("== 7 flat transfer")
m=RamseyCassKoopmansModel(); hp=StateSpaceHyperParams(m;Nk=500); r=solve_HJB(m,hp;verbose=false,flat_transfer=0.2); sm=SolvedModel(m,r)
k=r.variables.k; kd=sm.kdot_function.(k); actual=r.variables.y.-m.δ.*k.-r.variables.c
@printf(" kdot_function error vs generator drift = %.4f (transfer=0.2)\n",maximum(abs.(kd.-actual)))
println("== 8 V_err stochastic residual")
m=StochasticSkibaModel(); hp=StateSpaceHyperParams(m;Nk=200,Nz=10); r=solve_HJB(m,hp;verbose=false,crit=1e-10); 
ve=G.V_err(m)(r.value,r.variables); full=vec(r.variables.c.^(1-m.γ)./(1-m.γ)).+r.value.A*vec(r.value.v).-m.ρ.*vec(r.value.v)
@printf(" V_err max=%.3e  true residual max=%.3e\n",maximum(abs.(ve)),maximum(abs.(full)))
println("== 9 misc")
try; StateSpaceHyperParams(StochasticSkibaModel(PoissonProcess(z=[0.,.1],Q=[-.2 .2;.3 -.3]))); println(" poisson hp ok"); catch e; println(" poisson default hp: ",typeof(e)," ",sprint(showerror,e)); end
m=SkibaModel(); kk=2.2; e=1e-6
@printf(" Skiba f'(2.2)=%.5f FD=%.5f\n",only(G.production_function_prime(m,kk)),(only(G.production_function(m,kk+e))-only(G.production_function(m,kk-e)))/2e)
try; solve_HJB(RamseyCassKoopmansModel(γ=1.0),StateSpaceHyperParams(RamseyCassKoopmansModel();Nk=50);verbose=false); println(" gamma1 ran"); catch e; println(" gamma=1: ",sprint(showerror,e)); end
println(" grid coef=1 power=0 first=",G.generate_grid(5,1.0,2.0,1.0,0.0))
m=StochasticSkibaModel(); st=StateSpace(m,StateSpaceHyperParams(m;Nk=5,Nz=3)); println(" y[1],y[end] checks only; y at kmax,zlow=",st.aux_state.y[end,1])
# statespace constraint: low z at kmax
pr=OrnsteinUhlenbeckProcess(θ=.1,σ=.05);m=StochasticRamseyCassKoopmansModel(pr;δ=0.3,A=0.6);hpz=StateSpaceHyperParams((k=HyperParams(N=20,xmin=.01,xmax=1.0),z=HyperParams(N=3,xmin=.2,xmax=5.)));st=StateSpace(m,hpz)
c0=st.aux_state.y.-m.δ.*st[:k]; println(" min c0 in grid=",minimum(c0)," check passes=",try check_statespace_constraints(st,m); true catch; false end)
for γ in (2.0,1.5)
  m2=StochasticRamseyCassKoopmansModel(pr;δ=0.3,A=0.6,γ=γ)
  try; r=solve_HJB(m2,hpz,st;verbose=false); println(" gamma=$γ negative-c0 grid: conv=",r.value.convergence_status," min c=",r.variables===nothing ? nothing : minimum(r.variables.c)); catch e; println(" gamma=$γ negative-c0 grid threw ",typeof(e)); end
end
println("== 10 convergent evolution vs ODE (deterministic path)")
m=RamseyCassKoopmansModel(); kss=G.k_steady_state(m); hp=StateSpaceHyperParams((k=HyperParams(N=2000,xmin=.2kss,xmax=1.3kss),));r=solve_HJB(m,hp;verbose=false,crit=1e-9);sm=SolvedModel(m,r);k=r.variables.k
w=create_dx_stacked(k,1);i0=argmin(abs.(k.-0.5kss));g=zeros(2000);g[i0]=1/w[i0]
drift=r.variables.y.-m.δ.*k.-r.variables.c
fdrift(x)=begin j=clamp(searchsortedlast(k,x),1,1999);t=(x-k[j])/(k[j+1]-k[j]);(1-t)*drift[j]+t*drift[j+1] end
function rk(x,T;h=.01) for _ in 1:round(Int,T/h); a=fdrift(x);b=fdrift(x+h/2*a);c=fdrift(x+h/2*b);d=fdrift(x+h*c);x+=h/6*(a+2b+2c+d);end;x end
for (dt) in (nothing,1.0,0.1,0.01)
  ev=StateEvolution(g,sm,[0.0,10.0,40.0];dt=dt)
  ms=[dot(k,ev.cell_masses[:,i]) for i in 1:3]; v=[sqrt(max(dot(k.^2,ev.cell_masses[:,i])-ms[i]^2,0)) for i in 1:3]
  println(" dt=$dt mean k(10)=",round(ms[2],digits=4)," sd=",round(v[2],digits=4)," mean k(40)=",round(ms[3],digits=4)," sd=",round(v[3],digits=4))
end
println(" ODE k(10)=",round(rk(k[i0],10),digits=4)," k(40)=",round(rk(k[i0],40),digits=4))
println("== 11 ForwardDiff vs FD")
hpd=StateSpaceHyperParams((k=HyperParams(N=200,xmin=.1,xmax=8.),))
fdet(p)=begin m=RamseyCassKoopmansModel(2.0,.3,p,.05,.6); r=solve_HJB(m,hpd;verbose=false,crit=1e-11); r.value.v[100] end
fsk(p)=begin m=SkibaModel(2.0,.3,.05,.05,.6,.4,p); r=solve_HJB(m,hpd;verbose=false,crit=1e-11); r.value.v[100] end
ppoi=p->PoissonProcess(z=[p*one(p),2p],Q=[-.2 .2;.3 -.3]*one(p))
hps=StateSpaceHyperParams((k=HyperParams(N=100,xmin=.1,xmax=8.),z=HyperParams(N=2,xmin=.1,xmax=.2)))
fst(p)=begin m=StochasticSkibaModel(ppoi(p);γ=2one(p),α=.3one(p),ρ=.05one(p),δ=.05one(p),A_H=.6one(p),A_L=.4one(p),κ=2one(p)); r=solve_HJB(m,hps;verbose=false,crit=1e-11); r.value.v[50,2] end
hpa=StateSpaceHyperParams((k=HyperParams(N=60,xmin=.1,xmax=8.),z=HyperParams(N=2,xmin=.1,xmax=.2),η=HyperParams(N=2,xmin=.9,xmax=1.1)))
fab(p)=begin m=StochasticSkibaAbilityModel(ppoi(p);γ=2one(p),α=.3one(p),ρ=.05one(p),δ=.05one(p),A_H=.6one(p),A_L=.4one(p),κ=2one(p),η_mean=one(p)); r=solve_HJB(m,hpa;verbose=false,crit=1e-11); r.value.v[30,2,2] end
for (n,f,x) in (("RCK d/dρ",fdet,.05),("Skiba d/dκ",fsk,2.0),("StochSkiba Poisson",fst,.1),("Ability",fab,.1))
  try
   d=ForwardDiff.derivative(f,x); h=1e-5; fd=(f(x+h)-f(x-h))/2h; @printf(" %-20s AD=%.6f FD=%.6f\n",n,d,fd)
  catch e; println(" $n threw: ",sprint(showerror,e)[1:min(end,200)]); end
end
println("== 12 Adjoint update cost")
n=3000; d=fill(1.0,n-1); A=spdiagm(-1=>d,1=>d,0=>[-1;fill(-2.0,n-2);-1]); ws=EvolutionWorkspace(sparse(A'),ones(n))
@printf(" update with sparse: %.4fs\n",@elapsed update_evolution!(ws,sparse(A')))
@printf(" update with Adjoint: %.4fs\n",@elapsed update_evolution!(ws,A'))
