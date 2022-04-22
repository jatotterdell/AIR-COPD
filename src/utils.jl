function sampling(n::Int, p::AbstractWeights)
    randomise(MassWeightedUrn(p, 2), n)
end


function randomise_new(
    n::Int,
    p::AbstractWeights,
    n_current::Vector{Int} = zeros(Int, length(p)),
    n_max::Int = Inf,
)
    n_rem = max.(0, n_max .- n_current) .* (p .> 0)
    n_tot = sum(n_rem)
    if n_tot <= n # In this case, recruitment will cover all remaining participants
        n_new = n_rem
    else # Otherwise, residual sampling
        n_new = min.(n_rem, floor.(Int, n * p))
        n_rem = max.(n_max .- n_current .- n_new) .* (p .> 0)
        M = sum(n_new)
        R = n - M
        n_new += residual_sampling(R, p .* (n_rem .> 0))
    end
    return n_new
end


"""
    generate_data(ℙy::Vector{<:UnivariateDistribution}, n::Int, p::AbstractWeights)

Generate trial data.
`n` outcomes are generated from `ℙy` and treatment assignments from `p`.
"""
function generate_data(ℙy::Vector{<:UnivariateDistribution}, n::Int, p::AbstractWeights)
    x = sampling(n, p)
    y = rand.(ℙy[x])
    return (x, y)
end


function lposterior(θ, X, y, Π)
    β = θ[1:end-1]
    σ² = exp(θ[end])
    # If there's no data, just approximate the prior
    if isempty(X) | isempty(y)
        return sum(logpdf.(Π, [β; σ²]))
    else
        η = X * β
        return sum(logpdf.(Π, [β; σ²])) + sum(logpdf.(Normal.(η, sqrt(σ²)), y))
    end
end


"""
    decide(𝐏, 𝐃, ϵ0, ϵ1)

Trial decision function using current posterior probabilities `𝐏`, past decisions `𝐃` and 
decision thresholds `ϵ0` and `ϵ1`.
If a past decision has been made `𝐃 != 0`, then return that same decision, otherwise, assess for
futility, `𝐏 < ϵ0` or effectiveness `𝐏 > ϵ1`.
"""
function decide(𝐏, 𝐃, ϵ0, ϵ1)
    [q[2] != 0 ? q[2] : q[1] .< ϵ0 ? 1 : q[1] .> ϵ1 ? 2 : 0 for q in zip(𝐏, 𝐃)]
end





"""
    long_trial_DF(DF::DataFrame)

Transforms the result of `trial_DF` into long rather than wide format.
"""
function long_trial_DF(DF::DataFrame)
    fields = fieldnames(TrialResult)
    DFout = DataFrame[]
    for field in fields
        tmp = stack(DF, Regex(string(field) * "_"), [:trial, :interim])
        transform!(tmp, :variable => ByRow(x -> split(x, "_")) => [:variable, :arm])
        transform!(tmp, :arm => x -> parse.(Int, x), renamecols = false)
        push!(DFout, unstack(tmp, :variable, :value))
    end
    sort!(innerjoin(DFout..., on = [:trial, :interim, :arm]), [:trial, :interim, :arm])
end


"""
    summarise_trial(DF::DataFrame)

Summarise the results of a trial stored in a long `DataFrame` (i.e. from `long_trial_DF()`)
"""
function summarise_trial(DF::DataFrame)
    function meanmean(x)
        mean(mean.(x))
    end
    function meanscale(x)
        mean(scale.(x))
    end
    decision1(x) = mean(x .== 1)
    decision2(x) = mean(x .== 2)
    df1 = combine(groupby(DF, [:interim, :arm]), [:n, :ȳ, :𝐏] .=> [mean std])
    df2 = combine(groupby(DF, [:interim, :arm]), [:ℙθ, :ℙμ] .=> [meanmean meanscale])
    df3 = combine(groupby(DF, [:interim, :arm]), [:𝐃] .=> [decision1 decision2])
    innerjoin(df1, df2, df3, on = [:interim, :arm])
end