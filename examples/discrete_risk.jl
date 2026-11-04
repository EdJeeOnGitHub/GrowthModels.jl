# Run with: julia --project=. examples/discrete_risk.jl
using GrowthModels, Random

# Individual productivity risk. sigma is an innovation SD in log productivity.
shock=LogAR1(persistence=0.9,innovation_std=0.1,log_mean=0.0)
types=Population([0.8,1.2],[0.3,0.7])
model=DiscreteSkibaModel(productivity=shock,population=types)
grid=StateGrid(model;k=(n=200,bounds=(0.001,20.0)),z=(n=5,))
fit=solve(model,grid;history=true)
require_converged(fit)

# The matrix alternative uses the same model and solver interface:
explicit=MarkovChain(values=[0.8,1.2],transition=[0.9 0.1;0.2 0.8])
explicit_model=DiscreteSkibaModel(productivity=explicit,population=types)

initial_mass=zeros(size(grid))
i=argmin(abs.(grid.k .- 3.2))
initial_mass[i,3,:].=types.shares
initial=distribution(grid;masses=initial_mass)
evolution=Evolution(fit)
path=trajectory(evolution,initial,0:50)
long_run=stationary(fit,initial) # Includes initial-dependent absorption weights.
individual=simulate(fit,(k=grid.k[i],z=grid.z[3],ability=0.8),50;
    rng=MersenneTwister(42))
println((mean_capital=expectation(path,s->s.k)[end],
    stationary_mean=expectation(long_run,s->s.k),
    ability_shares=vec(sum(mass(final(path));dims=(1,2))),
    individual_final=last(individual.states)))
