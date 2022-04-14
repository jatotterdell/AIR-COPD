"""
    multinomial_sampling(n::Int, p::AbstractVector)

Use multinomial resampling to draw `n` integers in `{1,...,length(p)}` according to probability `p`.
"""
function multinomial_sampling(n::Int, p::AbstractVector)
    rand(Multinomial(n, p / sum(p)))
end


"""
    residual_sampling(n::Int, p::AbstractVector)

Use residual resampling to draw `n` integers in `{1,...,length(p)}` according to probability `p`.
"""
function residual_sampling(n::Int, p::AbstractVector)
    q = p / sum(p)
    n_new = floor.(Int, n * q)
    R = n - sum(n_new)
    n_new += multinomial_sampling(R, q)
    return n_new
end
