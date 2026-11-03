using GrowthModels, LinearAlgebra, SparseArrays, ForwardDiff, Printf, Statistics
const G = GrowthModels
println("== B AD vs FD (dual grid)")
function hpd(o) StateSpaceHyperParams((k=HyperParams(N=200,xmin=.1o,xmax=8o),)) end
hps(o)=StateSpaceHyperParams((k=HyperParams(N=100,xmin=.1o,xmax=8o),z=HyperParams(N=2,xmin=.1o,xmax=.2o)))
hpa(o)=StateSpaceHyperParams((k=HyperParams(N=60,xmin=.1o,xmax=8o),z=HyperParams(N=2,xmin=.1o,xmax=.2o),η=HyperParams(N=2,xmin=.9o,xmax=1.1o)))
hpo(o)=StateSpaceHyperParams((k=HyperParams(N=60,xmin=.1o,xmax=8o),z=HyperParams(N=6,xmin=.8o,xmax=1.2o)))
Q=[-.2 .2;.3 -.3]; pp(o)=PoissonProcess(z=[0.1o,0.2o],Q=Q.*o)
kw=(verbose=false,crit=1e-11)
cases=[("RCK dv/dρ",p->(o=one(p);solve_HJB(RamseyCassKoopmansModel(2o,.3o,p,.05o,.6o),hpd(o);kw...).value.v[100]),.05),
("Skiba dv/dκ",p->(o=one(p);solve_HJB(SkibaModel(2o,.3o,.05o,.05o,.6o,.4o,p),hpd(o);kw...).value.v[100]),2.0),
("Skiba dc/dκ",p->(o=one(p);solve_HJB(SkibaModel(2o,.3o,.05o,.05o,.6o,.4o,p),hpd(o);kw...).variables.c[100]),2.0),
("Poisson dv/dγ",p->(o=one(p);solve_HJB(StochasticSkibaModel(pp(o);γ=p,α=.3o,ρ=.05o,δ=.05o,A_H=.6o,A_L=.4o,κ=2o),hps(o);kw...).value.v[50,2]),2.0),
("OU dv/dσ",p->(o=one(p);solve_HJB(StochasticSkibaModel(OrnsteinUhlenbeckProcess(θ=.1o,σ=p);γ=2o,α=.3o,ρ=.05o,δ=.05o,A_H=.6o,A_L=.4o,κ=2o),hpo(o);kw...).value.v[30,3]),.1),
("Ability dv/dA_H",p->(o=one(p);solve_HJB(StochasticSkibaAbilityModel(pp(o);γ=2o,α=.3o,ρ=.05o,δ=.05o,A_H=p,A_L=.4o,κ=2o,η_mean=o),hpa(o);kw...).value.v[30,2,2]),.6),
("Evolution dE[k]/dρ",p->(o=one(p);m=RamseyCassKoopmansModel(2o,.3o,p,.05o,.6o);h=hpd(o);r=solve_HJB(m,h;kw...);sm=SolvedModel(m,r);w=create_dx_stacked(r.variables.k,1);ev=StateEvolution(ones(200)./sum(ForwardDiff.value.(w)),sm,[0.0,5.0];dt=1.0);dot(r.variables.k,ev.cell_masses[:,2])),.05),
("Stationary dE[k]/dσ",p->(o=one(p);m=StochasticSkibaModel(OrnsteinUhlenbeckProcess(θ=.1o,σ=p);γ=2o,α=.3o,ρ=.05o,δ=.05o,A_H=.6o,A_L=.4o,κ=2o);h=hpo(o);r=solve_HJB(m,h;kw...);sm=SolvedModel(m,r);s=StationaryDistribution(sm;initial_density=ones(360));w=create_dx_stacked(r.variables.k[:,1],6);dot(vec(r.variables.k),w.*s)),.1)]
for (n,f,x) in cases
 try; d=ForwardDiff.derivative(f,x);h=1e-5;fd=(f(x+h)-f(x-h))/2h;@printf(" %-22s AD=%.6f FD=%.6f rel=%.1e\n",n,d,fd,abs(d-fd)/max(abs(fd),1e-12)); catch err; println(" $n threw: ",first(sprint(showerror,err),220)); end
end
println("== D default dt dependence")
let m=StochasticSkibaModel(); hp=StateSpaceHyperParams(m;Nk=300,Nz=10); r=solve_HJB(m,hp;verbose=false); sm=SolvedModel(m,r); N=length(r.value.v)
  k=vec(r.variables.k); g=ones(N)
  for ts in ([0.,20.],[0.,5.,10.,20.],collect(0.:1.:20.),collect(0.:.1:20.))
    ev=StateEvolution(g,sm,ts;save=:final); @printf(" %3d obs: E[k(20)]=%.5f sd=%.5f\n",length(ts),dot(k,vec(ev.cell_masses)),sqrt(dot(k.^2,vec(ev.cell_masses))-dot(k,vec(ev.cell_masses))^2))
  end
end
println("== E stationary normalize=false")
let A=sparse([-1.0 1;1 -1]);w=[.5,2.];ev=StateEvolution([4.0,0.],A,[0.,1.],(2,),w;normalize=false,stationary=true)
 println(" final mass=",weighted_mass(ev.final_density,w)," stationary mass=",weighted_mass(ev.stationary_dist,w)); end
println("== F coupled third state")
let k=[1.,2.,3.];z=[1.,2.];η=[1.,2.];y=ones(3,2,2).+.05 .*reshape(k,3,1,1)
 m=StochasticSkibaAbilityModel(PoissonProcess(z=z,Q=[-.1 .1;.1 -.1]));hp=StateSpaceHyperParams((k=HyperParams(N=3,xmin=1.,xmax=3.),z=HyperParams(N=2,xmin=1.,xmax=2.),η=HyperParams(N=2,xmin=1.,xmax=2.)));st=StateSpace((k=k,z=z,η=η),(y=y,))
 B=kron(sparse([-.1 .1;.1 -.1]),sparse(I,6,6))*1.0
 try; update_v(m,Value(st),st,hp,B;verbose=false); println(" :auto ok"); catch err; println(" :auto: ",first(sprint(showerror,err),120)); end
 try; update_v(m,Value(st),st,hp,B;verbose=false,linear_solver=:sparse); println(" :sparse ok"); catch err; println(" :sparse threw ",first(sprint(showerror,err),120)); end
end
println("== G non-converged API")
let m=SkibaModel();hp=StateSpaceHyperParams(m;Nk=100);r=solve_HJB(m,hp;verbose=false,maxit=2)
 println(" variables=",r.variables); try; SolvedModel(m,r); catch err; println(" SolvedModel: ",typeof(err)); end; end
println("== H smooth skiba steep β")
println(" prime at β=500,k=.01: ",G.smooth_skiba_production_function_prime(.01,.3,.6,.4,2.0,500.0))
println("== I evolve-factor overhead")
let A=sparse([-1.0 1;1 -1]);w=[1.,1.];ws=EvolutionWorkspace(A,w;time_step=.3);StateEvolution([1.,0.],A,[0.,1.],(2,),w;dt=.3,workspace=ws);println(" factorizations for one fixed step size dt=.3 (actual .25): ",ws.factorizations-1)
 ws=EvolutionWorkspace(A,w;time_step=1.);StateEvolution([1.,0.],A,[0.,1.,3.,7.,15.],(2,),w;workspace=ws);println(" one-step-per-interval, doubling grid: ",ws.factorizations-1)
end
