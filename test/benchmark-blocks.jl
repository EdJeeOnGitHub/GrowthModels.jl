# Compares block solves across checkouts and thread counts. One case per process, so
# peak memory is per case:
# julia --project=/path/to/checkout -t 4 test/benchmark-blocks.jl single 200 21 20 7
# Scenarios: single Nk Nz Na samples | oneblock_det | oneblock_ou | nested | nested_ad
#            prev_warm | prev_warm_nested
using GrowthModels, LinearAlgebra, Printf, Statistics
const GM = GrowthModels
const FD = GM.ForwardDiff
BLAS.set_num_threads(1)
pop(n) = Population(collect(range(.8,1.2;length=n)),fill(1/n,n))
base = StochasticSkibaAbilityModel()
function timed(f; samples=5)
    f(); ts=Float64[]; r=nothing
    for _ in 1:samples; GC.gc(); t0=time(); r=f(); push!(ts,time()-t0); end
    median(ts), minimum(ts), r
end
scenario = ARGS[1]
label, (med, mn, conv) = if scenario == "single"
    Nk,Nz,Na,s = parse.(Int,ARGS[2:5])
    m=ResourceModel(base;population=pop(Na)); g=StateGrid(m;k=(n=Nk,),z=(n=Nz,))
    a,b,r=timed(()->solve(m,g);samples=s); "$(Nk)x$(Nz)x$(Na)",(a,b,r.converged)
elseif scenario == "oneblock_det"
    m=SkibaModel(); g=StateGrid(m;k=(n=1000,bounds=(.01,8.)))
    a,b,r=timed(()->solve(m,g);samples=25); "skiba_det_1000",(a,b,r.converged)
elseif scenario == "oneblock_ou"
    m=StochasticSkibaModel(); g=StateGrid(m;k=(n=200,bounds=(.01,8.)),z=(n=20,bounds=(.5,1.5)))
    a,b,r=timed(()->solve(m,g);samples=9); "skiba_ou_200x20",(a,b,r.converged)
elseif scenario == "nested"
    m0=ResourceModel(base;population=pop(10)); g=StateGrid(m0;k=(n=200,),z=(n=21,))
    ms=[ResourceModel(base;transfer=x,population=pop(10)) for x in range(0.,.2;length=12)]
    f=()->begin out=Vector{Any}(undef,12); Threads.@threads :dynamic for i in 1:12; out[i]=solve(ms[i],g); end; all(x->x.converged,out) end
    a,b,r=timed(f;samples=3); "nested_12x200x21x10",(a,b,r)
elseif scenario == "nested_ad"
    m0=ResourceModel(base;population=pop(5)); g=StateGrid(m0;k=(n=100,),z=(n=11,))
    grad(x)=FD.derivative(y->sum(solve(ResourceModel(base;transfer=y,population=pop(5)),g).value),x)
    f=()->begin out=zeros(8); xs=range(0.,.2;length=8); Threads.@threads :dynamic for i in 1:8; out[i]=grad(xs[i]); end; all(isfinite,out) end
    a,b,r=timed(f;samples=3); "nested_ad_8x100x11x5",(a,b,r)
elseif scenario == "prev_warm"
    m0=ResourceModel(base;population=pop(10)); g=StateGrid(m0;k=(n=200,),z=(n=21,))
    ms=[ResourceModel(base;transfer=x,population=pop(10)) for x in range(0.,.2;length=12)]
    f=()->begin fit=solve(ms[1],g); ok=fit.converged; for i in 2:12; fit=solve(ms[i],g;initial=fit); ok&=fit.converged; end; ok end
    a,b,r=timed(f;samples=3); "prev_warm_12x200x21x10",(a,b,r)
elseif scenario == "prev_warm_nested"
    m0=ResourceModel(base;population=pop(10)); g=StateGrid(m0;k=(n=200,),z=(n=21,))
    ms=[ResourceModel(base;transfer=x,population=pop(10)) for x in range(0.,.2;length=12)]
    f=()->begin ok=fill(true,4); Threads.@threads :dynamic for c in 1:4
        fit=solve(ms[3c-2],g); ok[c]&=fit.converged; for i in 3c-1:3c; fit=solve(ms[i],g;initial=fit); ok[c]&=fit.converged; end; end; all(ok) end
    a,b,r=timed(f;samples=3); "prev_warm_nested_4x3",(a,b,r)
end
@printf("%s,%d,%.4f,%.4f,%d,%s\n",label,Threads.nthreads(),med,mn,Sys.maxrss()÷2^20,conv)
