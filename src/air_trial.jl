"""
    AIRTrialParameters

Specifies the trial parameters.
Default arguments are provided.
# Fields
- ℙy: generative data model
- Π: model parameter priors 
- X: design matrix 
- nseq: sequence of sample sizes
- nmax: the maximum sample size allowed on any one arm
- p: initial target allocation to arms 
- ϵ0: bound for failure
- ϵ1: bound for graduate
- method: method of estimating the trial model
# Functions
Accessors for `AIRTrialParameters`
- `Xdesign`: get the design matrix
- `prior`: get the prior
- `drop_ϵ`: get the drop arm threshold 
- `grad_ϵ`: get the graduate arm threshold
"""
@with_kw struct AIRTrialParameters
    ℙy::Vector{Normal{Float64}} = Normal.([0.5, 0.5, 0.5, 0.5], 2.0)
    Π::Vector{<:UnivariateDistribution} = [
        Normal(0.5, 0.5)
        Normal.(zeros(3), 1)
        InverseGamma(1.5, 6)
    ]
    X::Matrix{Float64} = hcat(ones(4), vcat(zeros(3)', diagm(ones(3))))
    nseq::AbstractVector = 100:50:200
    nmax::Int = 50
    p::AbstractWeights = weights([0.25 for _ = 1:4])
    ϵ0::Float64 = 0.1
    ϵ1::Float64 = 0.9
    method::String = "Laplace"
end

DrWatson.default_prefix(T::AIRTrialParameters) = "Trial"
DrWatson.default_allowed(::AIRTrialParameters) =
    (Real, Vector{<:Real}, AbstractVector{Int}, String)


"""
    AIRTrialParameters(d::Dict)

Convert a `Dict` into an instance of `AIRTrialParameters` using keyword arguments.
"""
function AIRTrialParameters(d::Dict)
    args = (; (Symbol(k) => v for (k, v) in d)...)
    AIRTrialParameters(; args...)
end

Xdesign(p::AIRTrialParameters) = p.X
prior(p::AIRTrialParameters) = p.Π
drop_ϵ(p::AIRTrialParameters) = p.ϵ0
grad_ϵ(p::AIRTrialParameters) = p.ϵ1


"""
    TrialData

Mutable type for state of trial data.
# Functions
- `getx` returns the x vector 
- `gety` returns the y vector
- `samplesize` returns the total sample size
- `stats` returns the group counts and means
- `newdata(data, x, y)` will append new data and return a new `TrialData` object
"""
struct TrialData
    x::Vector{Int8}
    y::Vector{Float64}
end
TrialData() = TrialData(Int8[], Float64[])

function Base.show(io::IO, D::TrialData)
    n, y = stats(D)
    println(io, "n = $n")
    println(io, "ȳ = $(round.(y, digits = 2))")
end

getx(data::TrialData) = data.x
gety(data::TrialData) = data.y
counts(data::TrialData) =
    isempty(data.x) ? Int[] : [count(==(z), data.x) for z = 1:maximum(data.x)]
means(data::TrialData) =
    isempty(data.y) ? Float64[] : [mean(data.y[data.x.==z]) for z = 1:maximum(data.x)]
samplesize(data::TrialData) = length(data.x)


function stats(data::TrialData)
    n = counts(data)
    ȳ = means(data)
    if length(n) != length(ȳ)
        error("Something went wrong, length(n) != length(ȳ)")
    end
    return (n = n, ȳ = ȳ)
end


function newdata(
    data::TrialData,
    x::Union{Int8,Vector{Int8}},
    y::Union{Float64,Vector{Float64}},
)
    if length(x) != length(y)
        error("x and y must have same length")
    end
    return TrialData(vcat(data.x, Int8.(x)), vcat(data.y, y))
end



"""
    TrialArms(n::Int)
    TrialArms(params::AIRTrialParameters)

Type for state of trial arms.
Stores information on arm target allocations, decision probabilities, and made decisions.
Decisions are perpetual, e.g. if a decision has previously been made it carries forward.
# Functions
- `target`: return the current target allocations
- `decprob`: return the current decision probabilities 
- `decision`: return decisions which have  been made
"""
struct TrialArms
    target::AbstractWeights
    decprob::Vector{Float64}
    decision::Vector{Int8}
end
TrialArms(n::Int) =
    TrialArms(weights([1 / n for _ = 1:n]), [NaN for _ = 1:n], zeros(Int8, n))
TrialArms(params::AIRTrialParameters) =
    TrialArms(params.p, [NaN for _ = 1:length(params.p)], zeros(Int8, length(params.p)))
target(arms::TrialArms) = arms.target
decprob(arms::TrialArms) = arms.decprob
decision(arms::TrialArms) = arms.decision


"""
    TrialState

Keeps track of the current state of the trial.

Can be initalised with a `AIRTrialParameters` which sets the data to empty and next analysis to 1.
The model will be reflective of the prior.

`TrialState` is immutable, so `step` returns a new instance of `TrialState` rather 
than mutating the current state.

----

# Functions 
- `next_analysis`: return next scheduled analysis
- `counts`: return group specific sample sizes 
- `means`: return group specific means
- `new_data`: generate new data given the current state
- `step`: step through the next scheduled analysis
"""
struct TrialState
    params::AIRTrialParameters
    next_analysis::Int8
    data::TrialData
    arms::TrialArms
    posterior::MvNormal{Float64}
    termination::Int8
end
function TrialState(params::AIRTrialParameters)
    data = TrialData()
    posterior = MvNormal(
        LaplaceApproximation(
            lposterior,
            Xdesign(params)[data.x, :],
            data.y,
            prior(params),
        )...,
    )
    arms = TrialArms(params)
    TrialState(params, one(Int8), data, arms, posterior, zero(Int8))
end

function Base.show(io::IO, S::TrialState)
    n, y = stats(S.data)
    println(io, "Analyses: $(next_analysis(S) - one(typeof(next_analysis(S))))")
    println(io, "TrialData:")
    println(io, "  n = $n")
    println(io, "  ȳ = $(round.(y, digits = 2))")
    println(io, "TrialArms:")
    println(io, "  target = $(target(S.arms))")
    println(io, "  decision = $(decision(S.arms))")
    print(io, "Termination: $(S.termination)")
end

next_analysis(state::TrialState) = state.next_analysis

counts(state::TrialState) = counts(state.data)
means(state::TrialState) = means(state.data)


"""
    new_data(state::TrialState)

Update `TrialData` from the current `TrialState`
"""
function new_data(state::TrialState)
    analysis = next_analysis(state)
    if analysis == one(Int8)
        n_new = state.params.nseq[1]
    else
        n_new = state.params.nseq[analysis] - state.params.nseq[analysis-one(Int8)]
    end
    x_new, y_new = generate_data(state.params.ℙy, n_new, state.arms.target)
    return newdata(state.data, Int8.(x_new), y_new)
end


"""
    new_posterior(params::AIRTrialParameters, data::TrialData)

Update model posterior from trial parameters and trial data.
"""
function new_posterior(params::AIRTrialParameters, data::TrialData)
    return MvNormal(
        LaplaceApproximation(
            lposterior,
            Xdesign(params)[getx(data), :],
            gety(data),
            prior(params),
        )...,
    )
end


"""
    new_arms(params::AIRTrialParameters, arms::TrialArms, posterior)

Update trial arms given trial parameters, previous arms, and new posterior.
"""
function new_arms(params::AIRTrialParameters, arms::TrialArms, posterior)
    decprob = decision_prob(marginal(posterior))
    int_dec = [
        q[2] != 0 ? q[2] : q[1] < drop_ϵ(params) ? 1 : q[1] > grad_ϵ(params) ? 2 : 0 for
        q in zip(decprob, decision(arms)[2:end])
    ]
    dec = [0; int_dec]
    new_target = weights(normalize(target(arms) .* (dec .== 0), 1))
    return TrialArms(new_target, decprob, dec)
end



"""
    summary(state::TrialState)

Return a summary tuple of `TrialState`.
Including group sample sizes and means, model parameter posteriors, and decisions.
"""
function summary(state::TrialState)
    n, ȳ = stats(state.data)
    ℙθ = marginal(state.posterior) # Model parameters
    ℙμ = marginal(Xdesign(state.params) * state.posterior) # Group means
    𝐏 = state.arms.decprob
    𝐃 = state.arms.decision
    return (n = n, ȳ = ȳ, ℙθ = ℙθ, ℙμ = ℙμ, 𝐏 = 𝐏, 𝐃 = 𝐃)
end



"""
    step(state::TrialState)

Perform the next analysis given current `TrialState`.
Used to "step" through interims.
"""
function step(state::TrialState)

    params = state.params
    analysis = AIR.next_analysis(state)

    # Data updates
    data = new_data(state)

    # Model update
    posterior = new_posterior(params, data)

    # Trial arm updates
    arms = new_arms(params, state.arms, posterior)

    # Termination
    termination =
        samplesize(data) == maximum(params.nseq) ? Int8(1) :
        all(decision(arms)[2:end] .!= 0) ? Int8(2) : Int8(0)

    # New Trial state
    return TrialState(
        state.params,
        analysis + one(Int8),
        data,
        arms,
        posterior,
        termination,
    )
end



"""
    TrialResult

A `TrialResult` type has fields for the parameter posteriors, `ℙθ`, and 
fields for the posterior probabilities used for decision making, `𝐏`, and 
fields for the decisions made, `𝐃` where:
- `0` means no decision
- `1` means dropped
- `2` means graduation
"""
struct TrialResult
    n::Matrix{Int}
    ȳ::Matrix{Float64}
    ℙθ::Matrix{Normal{Float64}} # Parameter posterior (Laplace approximation)
    ℙμ::Matrix{Normal{Float64}}  # Parameter posterior for cell means
    𝐏::Matrix{Float64} # Posterior probability for effectiveness
    𝐃::Matrix{Int} # Decision (effectiveness, futility)
end
function Base.show(io::IO, ::MIME"text/plain", T::TrialResult)
    println(io, "TrialResult:")
    print(io, "  Interims = $(size(T.𝐏, 1))")
end


"""
    TrialResult(S::Vector{TrialState})

Convert a vector of `TrialState`'s into a `TrialResult` object.
"""
function TrialResult(S::Vector{TrialState})
    N = length(S)
    K = length(S[end].arms.target)
    n = zeros(Int, N, K)
    ȳ = zeros(Float64, N, K)
    ℙθ = Matrix{Normal{Float64}}(undef, N, K)
    ℙμ = Matrix{Normal{Float64}}(undef, N, K)
    𝐏 = zeros(N, K)
    𝐏[:, 1] .= NaN
    𝐃 = zeros(Int, N, K)
    for i = 1:N
        n[i, :], ȳ[i, :], ℙθ[i, :], ℙμ[i, :], 𝐏[i, 2:end], 𝐃[i, :] = AIR.summary(S[i])
    end
    return TrialResult(n, ȳ, ℙθ, ℙμ, 𝐏, 𝐃)
end


"""
    trial_DF(res::Vector{TrialResult})

Hack function to make a `DataFrame` out of a vector of trial results.
"""
function trial_DF(res::Vector{TrialResult})
    fields = fieldnames(TrialResult)
    DF = DataFrame[]
    for field in fields
        push!(
            DF,
            vcat(
                [
                    (
                        f = getfield(x, field);
                        hcat(
                            DataFrame(:interim => 1:size(f, 1)),
                            DataFrame(
                                f,
                                Symbol.(string(field) .* "_" .* string.(1:size(f, 2))),
                            ),
                        )
                    ) for x in res
                ]...,
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
    simulate(T::AIRTrialParameters)

Simulate an AIR trial under parameters `T`.

There is no cap on sample sizes to active arms, instead 
stopping only occurs at the maximum sample size.
"""
function simulate(T::AIRTrialParameters)
    @unpack ℙy, Π, X, nseq, nmax, p, ϵ0, ϵ1 = T
    K = length(ℙy)
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
        # Data generation and storage
        x_new, y_new = generate_data(ℙy, n_new[interim], p_cur)
        x = vcat(x, x_new)
        y = vcat(y, y_new)
        n[interim, :] = [count(==(z), x) for z = 1:4]
        ȳ[interim, :] = [mean(y[x.==z]) for z = 1:4]
        # Model approximation
        μ, Σ = LaplaceApproximation(lposterior, X[x, :], y, Π)
        M = MvNormal(μ, Σ)
        # Outputs
        ℙμ[interim, :] = marginal(X * M)
        ℙθ[interim, :] = marginal(M)
        𝐏[interim, :] = [NaN; cdf.(ℙθ[interim, 2:end], 0)]
        # Decisions
        𝐃[interim, :] =
            interim == 1 ? decide(𝐏[interim, :], zeros(Int, K), ϵ0, ϵ1) :
            decide(𝐏[interim, :], 𝐃[interim-1, :], ϵ0, ϵ1)
        # Update target allocations
        p_cur = weights(normalize(p_cur .* [1; (𝐃[interim, 2:end] .== 0)], 1))
        if all(𝐃[interim, 2:end] .!= 0)
            break
        end
    end
    return TrialResult(
        n[1:interims, :],
        ȳ[1:interims, :],
        ℙθ[1:interims, :],
        ℙμ[1:interims, :],
        𝐏[1:interims, :],
        𝐃[1:interims, :],
    )
end



"""
    simulate2(T::AIRTrialParameters)

Simulate using `step` and `TrialState`
"""
function simulate2(T::AIRTrialParameters)
    N = length(T.nseq)
    S = Vector{TrialState}(undef, N + 1)
    S[1] = TrialState(T)
    final = N
    for interim = 1:N
        S[interim+1] = step(S[interim])
        if S[interim+1].termination != 0
            final = interim
            break
        end
    end
    return TrialResult(S[2:(final+1)])
end


"""
    run_sims_threads(id::Int, T::AIRTrialParameters, n::Int = 10_000)

Run the AIR-COPD simulation for `n` iterations under configuration `T` with label `id` for saving the results.
Results are saved in `datadir("sims")` with filename `sim_\$(id)`.
"""
function run_sims_threads(
    id::Int,
    T::AIRTrialParameters,
    n::Int = 10_000;
    save::Bool = true,
)
    t = time()
    d = struct2dict(T)
    res = Vector{TrialResult}(undef, n)
    Threads.@threads for i = 1:n
        res[i] = simulate2(T)
    end
    out_wide = trial_DF(res)
    out_long = long_trial_DF(out_wide)
    d[:result] = out_long
    d[:time] = time() - t
    if save
        wsave(datadir("sims", "sim_$(lpad(string(id), 2, "0")).jld2"), d)
    end
    return out_long
end
