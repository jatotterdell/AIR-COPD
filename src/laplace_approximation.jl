"""
    LaplaceApproximation(lposterior::Function, X, y)

Calculates a Laplace approximation to a log posterior function `lposterior` using data `X` and `y`.
The function returns a `MvNormal` distribution representing the Laplace approximation.
The function `lposterior(θ, X, y)` has arguments for the parameter, `θ`, the design, `X`, and responses `y`.
This function only returns the posterior approximation for model coefficients, ignoring the variance parameter.
"""
function LaplaceApproximation(lposterior::Function, X, y, Π)
    func = TwiceDifferentiable(
        θ -> -lposterior(θ, X, y, Π),
        ones(size(X, 2) + 1);
        autodiff = :forward,
    )
    opt = optimize(func, zeros(size(X, 2) + 1), LBFGS())
    est = Optim.minimizer(opt)
    est_hess = Optim.hessian!(func, est)
    var_cov = Hermitian(inv(est_hess))
    # don't care about the variance parameter
    return est[1:end-1], var_cov[1:end-1, 1:end-1]
end


"""
    marginal(M::MvNormal)

Extract the univariate Normal marginals from a multivariate Normal distribution.
"""
function marginal(M::MvNormal)
    return Normal.(params(M)[1], sqrt.(diag(params(M)[2])))
end


"""
    decision_prob(M::Vector{Normal{Float64}})

Calculate the specified decision probability.
"""
function decision_prob(M::Vector{Normal{Float64}})
    return cdf.(M[2:end], 0)
end
