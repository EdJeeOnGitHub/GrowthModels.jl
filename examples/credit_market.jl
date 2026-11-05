# Stationary credit-market equilibria across collateral limits, with grid refinement.
# Run with: julia --project=. examples/credit_market.jl
using GrowthModels, Printf

base = StochasticSkibaModel(OrnsteinUhlenbeckProcess(θ=-log(0.9),σ=0.1))
makegrid(n,nz,hi) = StateGrid(base;k=(n=n,bounds=(0.05,hi),spacing=:stretched),z=(n=nz,))
grids = (baseline=makegrid(300,15,60.0),capital=makegrid(600,15,60.0),
    productivity=makegrid(300,31,60.0),bound=makegrid(424,15,120.0))

@printf("%-13s %5s %8s %7s %7s %7s %7s %9s %6s\n","grid","λ","r*","K","Y","credit","high","top mass","jumps")
for (name,grid) in pairs(grids), λ in (1.5,2.0,3.0,5.0,20.0)
    eq = credit_equilibrium(base,grid;λ=λ)
    isempty(eq.equilibria) && @printf("%-13s %5.1f no clearing rate\n",name,λ)
    for e in eq.equilibria
        m = CreditModel(base;λ=λ,r=e.rate)
        high = expectation(e.distribution,s -> capital_choice(m,s).technology == 2)
        @printf("%-13s %5.1f %8.5f %7.3f %7.3f %7.3f %7.3f %9.2e %6d\n",name,λ,e.rate,e.capital,e.output,
            e.borrowing/e.output,high,e.upper_mass,length(eq.discontinuities))
    end
    flush(stdout)
end
