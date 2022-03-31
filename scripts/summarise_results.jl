using DrWatson
quickactivate(@__DIR__, "AIR")

using Parameters, LinearAlgebra, Distributions, StatsBase, StatsModels, DataFrames, Optim
using AIR

results = collect_results(datadir("sims"))
res = results.result
out = summarise_trial.(res)
