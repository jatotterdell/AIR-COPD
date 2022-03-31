using DrWatson
quickactivate(@__DIR__, "AIR")

using Parameters, LinearAlgebra, Distributions, StatsBase, StatsModels, DataFrames, Optim
using AIR

# Pre-compile
run_sims_threads(9999, AIRTrialParameters(), 2)

# Specify all configurations to explore
allparams = Dict(
    "μ" => [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
    "σ" => 1.6,
    "nseq" => [100:40:180, 100:40:260],
    "ϵ0" => 0.1,
    "ϵ1" => 0.9,
)

# Expand all config combinations
dicts = dict_list(allparams)
configs = AIRTrialParameters.(dicts)

# Enumerate over combinations.
for (id, config) in enumerate(configs)
    run_sims_threads(id, config)
end
