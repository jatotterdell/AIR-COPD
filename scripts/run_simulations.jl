using DrWatson
quickactivate(@__DIR__, "AIR")

using Parameters, LinearAlgebra, Distributions, StatsBase, StatsModels, DataFrames, Optim
using AIR

T = AIRTrialParameters()

function run_sims_threads(id::Int, T::AIRTrialParameters, n::Int = 10_000)
    d = struct2dict(T)
    res = Vector{AIRTrialResult}(undef, n)
    Threads.@threads for i = 1:n
        res[i] = simulate(T)
    end
    out_wide = trial_DF(res)
    out_long = long_trial_DF(out_wide)
    d[:result] = out_long
    wsave(datadir("sims", "sim_$(id).jld2"), d)
    return out_long
end

# Specify all configurations to explore
allparams = Dict(
    "μ" => [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
    "σ" => 1.6,
    "nseq" => [100:40:180, 100:40:260],
    "ϵ0" => 0.1,
    "ϵ1" => 0.9,
)
dicts = dict_list(allparams)
configs = AIRTrialParameters.(dicts)

for (id, config) in enumerate(configs)
    run_sims_threads(id, config);
end
