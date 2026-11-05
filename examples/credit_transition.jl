# Opening credit from autarky: the transition path at λ=2 and its robustness.
# Damping 0.6 did not converge within 200 iterations in testing, so it is not a case here.
# Run all cases with: julia --project=. examples/credit_transition.jl
# Or one case, by name, with: julia --project=. examples/credit_transition.jl dt
using GrowthModels, Printf

const CASES = (
    baseline=(n=200,nz=31,T=100.0,dt=0.25,damping=0.3,guess=nothing),
    horizon=(n=200,nz=31,T=150.0,dt=0.25,damping=0.3,guess=nothing),
    dt=(n=200,nz=31,T=100.0,dt=0.125,damping=0.3,guess=nothing),
    guess=(n=200,nz=31,T=100.0,dt=0.25,damping=0.3,guess=0.03),
    wealth=(n=400,nz=31,T=100.0,dt=0.25,damping=0.3,guess=nothing),
    productivity=(n=200,nz=61,T=100.0,dt=0.25,damping=0.3,guess=nothing),
    productivity2=(n=200,nz=121,T=100.0,dt=0.25,damping=0.3,guess=nothing),
)

function run_case(name,c)
    base = StochasticSkibaModel(OrnsteinUhlenbeckProcess(θ=-log(0.9),σ=0.1))
    grid = StateGrid(base;k=(n=c.n,bounds=(0.05,60.0),spacing=:stretched),z=(n=c.nz,))
    autarky = solve(base,grid)
    require_converged(autarky)
    start = stationary(autarky)
    terminal = credit_equilibrium(base,grid;λ=2.0,rates=range(0.03,0.045;length=6))
    N = round(Int,c.T/c.dt)
    guess = c.guess === nothing ? nothing : fill(c.guess,N)
    seconds = @elapsed path = credit_transition(base,grid,start;λ=2.0,terminal=terminal,T=c.T,dt=c.dt,
        damping=c.damping,initial_rates=guess,maxiters=200)
    d = path.diagnostics
    @printf("%s: %s, %.0fs\n",name,path,seconds)
    @printf("  autarky: Y=%.4f high=%.3f A=%.3f\n",expectation(start,s -> production(base,s)),
        expectation(start,s -> float(production(base,s) > 0 && base.A_H*max(s.k-base.κ,0)^base.α > base.A_L*s.k^base.α)),
        expectation(start,s -> s.k))
    @printf("  terminal: r*=%.5f; at T gaps r=%.1e A=%.1e TV=%.1e; upper mass=%.1e\n",clearing_rate(terminal),
        d.terminal.rate,d.terminal.wealth,d.terminal.total_variation,d.upper_mass)
    for t in (0.0,1.0,5.0,10.0,20.0,40.0,60.0,99.0)
        i = round(Int,t/c.dt)+1
        @printf("  t=%5.1f r=%.5f A=%.3f Y=%.4f high=%.3f credit/Y=%.3f ED/A=%+.1e\n",t,path.rates[i],path.wealth[i],
            path.output[i],path.high_technology[i],path.borrowing[i]/path.output[i],path.excess[i]/path.wealth[i])
    end
    flush(stdout)
end

for name in (isempty(ARGS) ? keys(CASES) : Symbol.(ARGS))
    run_case(name,CASES[name])
end
