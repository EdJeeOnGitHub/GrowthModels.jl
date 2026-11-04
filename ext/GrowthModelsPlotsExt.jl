module GrowthModelsPlotsExt
using GrowthModels, Plots
function GrowthModels.plot_model(fit::GrowthSolution)
    require_converged(fit)
    k = fit.grid.k
    value = reshape(fit.value,length(k),:)
    c = reshape(fit.policy.consumption,length(k),:)
    drift = reshape(fit.policy.drift,length(k),:)
    return plot(plot(k,value;xlabel="Capital",ylabel="Value",legend=false),
        plot(k,c;xlabel="Capital",ylabel="Consumption",legend=false),
        plot(k,drift;xlabel="Capital",ylabel="Capital drift",legend=false);layout=(1,3))
end
function GrowthModels.plot_diagnostics(fit::GrowthSolution)
    h = fit.diagnostics.history
    isempty(h) && throw(ArgumentError("No iteration history retained; solve with history=true"))
    return plot([x.iteration for x in h],hcat([x.update for x in h],[x.residual for x in h]);
        label=["Value update" "HJB residual"],yscale=:log10,xlabel="Iteration")
end
function GrowthModels.plot_model(fit::DiscreteGrowthSolution)
    require_converged(fit)
    k = fit.grid.k
    return plot(plot(k,reshape(fit.value,length(k),:);xlabel="Capital",ylabel="Value",legend=false),
        plot(k,reshape(fit.policy.consumption,length(k),:);xlabel="Capital",ylabel="Consumption",legend=false),
        plot(k,reshape(fit.policy.next_capital,length(k),:);xlabel="Capital",ylabel="Next-period capital",legend=false);
        layout=(1,3))
end
function GrowthModels.plot_diagnostics(fit::DiscreteGrowthSolution)
    h = fit.diagnostics.history
    isempty(h) && throw(ArgumentError("No iteration history retained; solve with history=true"))
    return plot([x.iteration for x in h],hcat([x.update for x in h],[x.residual for x in h]);
        label=["Value update" "Bellman residual"],yscale=:log10,xlabel="Iteration")
end

end
