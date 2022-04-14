using DrWatson
quickactivate(@__DIR__, "AIR")

using Parameters,
    LinearAlgebra, Distributions, StatsBase, StatsModels, DataFrames, Optim, ProgressMeter
using AIR

# Pre-compile
run_sims_threads(9999, AIRTrialParameters(), 2)

# Specify all configurations to explore
α = 0.5
β = [[0, 0, 0, 0], [0, -0.5, 0, 0], [0, -1, 0, 0]]
allparams = Dict(
    "μ" => [α .+ β[i] for i = 1:3],
    "σ" => [1, 1.5, 2],
    "nseq" => [100:50:200, 100:50:250],
    "ϵ0" => [0.1, 0.2],
    "ϵ1" => 0.9,
    "Π" => [
        [Normal(0.5, 0.5); Normal.(zeros(3), 1); InverseGamma(1.5, 6)],
        [Normal(0.5, 1.0); Normal.(zeros(3), 1); InverseGamma(1.5, 6)],
    ],
)

# Expand all config combinations
dicts = dict_list(allparams)
configs = AIRTrialParameters.(dicts)

# Enumerate over combinations.
@showprogress 1 "Running sims..." for (id, config) in enumerate(configs)
    run_sims_threads(id, config)
end
