module AIR

using DrWatson,
    LinearAlgebra, Parameters, Distributions, Printf, StatsBase, DataFrames, Optim, Printf
using RandomisationModels # For MassWeightedUrn

export AIRTrialParameters, AIRTrialResult
export LaplaceApproximation, marginal
export generate_data, lposterior, decide, simulate, trial_DF, long_trial_DF, summarise_trial

include(srcdir("laplace_approximation.jl"))
include(srcdir("air_trial.jl"))

end