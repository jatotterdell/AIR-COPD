module AIR

using DrWatson,
    LinearAlgebra, Parameters, Distributions, Printf, StatsBase, DataFrames, Optim, Printf

export AIRTrialParameters, AIRTrialResult
export LaplaceApproximation, marginal
export generate_data, lposterior, decide, simulate, trial_DF, long_trial_DF, summarise_trial

include("laplace_approximation.jl")
include("air_trial.jl")

end