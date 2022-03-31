"""
    AIRTrialParameters
"""
@with_kw struct AIRTrialParameters
    μ::Vector{<:Real} = [2.0, 2.0, 2.0, 2.0]
    σ::Real = 1.6
    Π::Vector{<:UnivariateDistribution} = [Normal.(0, [5, 5, 5, 5]); InverseGamma(1, 1)]
    X::Matrix{Float64} = hcat(ones(4), vcat(zeros(3)', diagm(ones(3))))
    nseq::AbstractVector = 100:40:180
    p::AbstractWeights = weights([0.25 for _ = 1:4])
    ϵ0::Float64 = 0.1
    ϵ1::Float64 = 0.9
    method::String = "Laplace"
end

function AIRTrialParameters(d::Dict)
    args = (; (Symbol(k) => v for (k, v) in d)...)
    AIRTrialParameters(; args...)
end


DrWatson.default_prefix(T::AIRTrialParameters) = "Trial"
DrWatson.default_allowed(::AIRTrialParameters) =
    (Real, Vector{<:Real}, AbstractVector{Int}, String)


"""
    AIRTrialResult

A `AIRTrialResult` type has fields for the parameter posteriors, `ℙθ`, and 
fields for the posterior probabilities used for decision making, `𝐏`, and 
fields for the decisions made, `𝐃` where:
    - `0` means no decision
    - `1` means futility
    - `2` means graduation (i.e. effectiveness)
"""
struct AIRTrialResult
    n::Matrix{Int}
    ȳ::Matrix{Float64}
    ℙθ::Matrix{Normal{Float64}} # Parameter posterior (Laplace approximation)
    ℙμ::Matrix{Normal{Float64}}  # Parameter posterior for cell means
    𝐏::Matrix{Float64} # Posterior probability for effectiveness
    𝐃::Matrix{Int} # Decision (effectiveness, futility)
end
Base.show(io::IO, ::MIME"text/plain", T::AIRTrialResult) =
    @printf(io, "AIRTrialResult(interims=%i)", size(T.𝐏, 1))


function generate_data(ℙy::Vector{<:UnivariateDistribution}, n::Int, p::AbstractWeights)
    # x = rand(Categorical(p), n)
    x = randomise(MassWeightedUrn(p, 4), n)
    y = rand.(ℙy[x])
    return (x, y)
end


function lposterior(θ, X, y, Π)
    β = θ[1:end-1]
    σ² = exp(θ[end])
    η = X * β
    return sum(logpdf.(Π, [β; σ²])) + sum(logpdf.(Normal.(η, sqrt(σ²)), y))
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


function simulate(T::AIRTrialParameters)
    @unpack μ, σ, Π, X, nseq, p, ϵ0, ϵ1 = T
    ℙy = Normal.(μ, σ)
    K = length(μ)
    N = length(nseq)
    # Outputs
    ℙθ = Matrix{Normal{Float64}}(undef, N, K)
    ℙμ = Matrix{Normal{Float64}}(undef, N, K)
    𝐏 = zeros(N, K)
    𝐃 = zeros(Int, N, K)
    x = zeros(Int, 0)
    y = zeros(Float64, 0)
    n = zeros(Int, N, K)
    ȳ = zeros(Float64, N, K)
    n_new = diff([0; nseq])
    p_cur = deepcopy(p)
    interims = 0
    for interim = 1:N
        interims += 1
        x_new, y_new = generate_data(ℙy, n_new[interim], p_cur)
        x = vcat(x, x_new)
        y = vcat(y, y_new)
        n[interim, :] = [count(==(z), x) for z = 1:4]
        ȳ[interim, :] = [mean(y[x.==z]) for z = 1:4]
        μ, Σ = LaplaceApproximation(lposterior, X[x, :], y, Π)
        M = MvNormal(μ[1:end-1], Σ[1:end-1, 1:end-1])
        ℙμ[interim, :] = marginal(X * M)
        ℙθ[interim, :] = marginal(M)
        𝐏[interim, :] = [NaN; cdf.(ℙθ[interim, 2:end], 0)]
        𝐃[interim, :] =
            interim == 1 ? decide(𝐏[interim, :], zeros(Int, K), ϵ0, ϵ1) :
            decide(𝐏[interim, :], 𝐃[interim-1, :], ϵ0, ϵ1)
        # 𝐃[interim, :] = [0; [q .< ϵ0 ? 1 : q .> ϵ1 ? 2 : 0 for q in 𝐏[interim, 2:end]]]
        p_cur = weights(normalize(p_cur .* [1; (𝐃[interim, 2:end] .== 0)], 1))
        if all(𝐃[interim, 2:end] .!= 0)
            break
        end
    end
    return AIRTrialResult(
        n[1:interims, :],
        ȳ[1:interims, :],
        ℙθ[1:interims, :],
        ℙμ[1:interims, :],
        𝐏[1:interims, :],
        𝐃[1:interims, :],
    )
end


"""

Hack function to make a DataFrame out of a vector of trial results.
"""
function trial_DF(res::Vector{AIRTrialResult})
    fields = fieldnames(AIRTrialResult)
    DF = DataFrame[]
    for field in fields
        push!(
            DF,
            vcat([(f = getfield(x, field);
            hcat(
                DataFrame(:interim => 1:size(f, 1)),
                DataFrame(f, Symbol.(string(field) .* "_" .* string.(1:size(f, 2)))))
                ) for x in res]...,
                source = :trial => 1:size(res, 1),
            ),
        )
    end
    return select!(
        innerjoin(DF..., on = [:interim, :trial]),
        :trial,
        :interim,
        Not([:trial, :interim]),
    )
end


"""

Transforms the result of `trial_DF` into long rather than wide.
"""
function long_trial_DF(DF::DataFrame)
    fields = fieldnames(AIRTrialResult)
    DFout = DataFrame[]
    for field in fields
        tmp = stack(DF, Regex(string(field) * "_"), [:trial, :interim])
        transform!(tmp, :variable => ByRow(x -> split(x, "_")) => [:variable, :arm])
        transform!(tmp, :arm => x -> parse.(Int, x), renamecols = false)
        push!(DFout, unstack(tmp, :variable, :value))
    end
    innerjoin(DFout..., on = [:trial, :interim, :arm])
end


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