module AIR

using DrWatson
using Reexport
@reexport using Distributions
using LinearAlgebra, Parameters, Printf, StatsBase, DataFrames, Optim, Printf
using RandomisationModels: MassWeightedUrn # For MassWeightedUrn

export AIRTrialParameters, AIRTrialResult
export LaplaceApproximation, marginal
export multinomial_sampling, residual_sampling
export generate_data,
    lposterior, decide, simulate, trial_DF, long_trial_DF, summarise_trial, run_sims_threads

include("laplace_approximation.jl")
include("resamplers.jl")
include("air_trial.jl")

end