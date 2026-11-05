module GrowthModels
using SparseArrays, LinearAlgebra, ForwardDiff, Random
using NaNMath: pow
import StatsBase: sample
import Base: show

abstract type Model{T<:Real} end
abstract type StochasticProcess end
abstract type DeterministicModel{T<:Real} <: Model{T} end
abstract type StochasticModel{T<:Real,S<:StochasticProcess} <: Model{T} end

export MarkovChain, LogAR1, transition, DiscreteEvolution, simulate,
    DiscreteTimeModel, DiscreteRamseyCassKoopmansModel, DiscreteSkibaModel,
    DiscreteGrowthSolution, next_capital, policy_jumps,
    Model, DeterministicModel, StochasticModel, StochasticProcess,
    SkibaModel, SmoothSkibaModel, RamseyCassKoopmansModel,
    StochasticSkibaModel, StochasticRamseyCassKoopmansModel,
    StochasticSkibaAbilityModel, StochasticNPAbilityModel,
    OrnsteinUhlenbeckProcess, PoissonProcess, from_stationary_OrnsteinUhlenbeckProcess, sample,
    Population, ResourceModel, FunctionalModel, population, production, resources,
    marginal_product, preferences, state_schema, default_grid_spec, generator, process, process_axis,
    StateGrid, cell_weights, coordinates, GrowthSolution, solve, regrid,
    require_converged, consumption, capital_drift, technology_crossing,
    StateDistribution, distribution, mass, density, expectation, boundary_mass,
    Evolution, ContinuousEvolution, trajectory, advance, observe, stationary, Trajectory, Observations,
    at, final, plot_model, plot_diagnostics,
    CreditModel, CreditEquilibrium, capital_choice, credit_equilibrium, clearing_rate,
    CreditTransition, credit_transition,
    DiscreteCreditModel, DiscreteCreditSolution, DiscreteCreditEquilibrium, credit_policy,
    discrete_credit_equilibrium, leverage

include("ReadOnly.jl")
include("StochasticProcesses.jl")
include("Models.jl")
include("LinearSolvers.jl")
include("HJB.jl")
include("Skiba.jl")
include("SmoothSkiba.jl")
include("RamseyCassKoopmans.jl")
include("StochasticRamseyCassKoopmans.jl")
include("StochasticSkiba.jl")
include("StochasticSkibaAbility.jl")
include("StochasticNPAbility.jl")
include("Economics.jl")
include("Grids.jl")
include("HJBWorkspace.jl")
include("BlockSolve.jl")
include("Solutions.jl")
include("DiscreteProcesses.jl")
include("DiscreteModels.jl")
include("Bellman.jl")
include("DiscreteSolutions.jl")
include("Distributions.jl")
include("StateEvolution.jl")
include("Evolution.jl")
include("CreditMarket.jl")
include("CreditTransition.jl")
include("DiscreteEvolution.jl")
include("DiscreteSimulation.jl")
include("DiscreteCredit.jl")

"""Plot a solution. Load Plots to activate the plotting extension."""
function plot_model end
"""Plot solver convergence diagnostics. Load Plots to activate the extension."""
function plot_diagnostics end
end
