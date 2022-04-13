using AIR
using Stan
using StanSample
using StatsBase
using LinearAlgebra

# Note that τ ∼ Half-t(ν, 0, s) ⟹ τ² ∼ InvGamma(a = ν/2, b = νs²/2)
# So if we think τ ∼ Half-t(3, 0, 2) then can assume τ² ∼ InvGamma(1.5, 6)

model = " 
data {
    int<lower=1> N;
    int<lower=1> P;
    vector[N] y;
    matrix[N, P] X;
}

parameters {
    real alpha;
    vector[P-1] beta;
    real<lower=0> sigmasq;
}
transformed parameters {
    vector[P] b;
    b[1] = alpha;
    b[2:P] = beta;
}
model {
    alpha ~ normal(1, 0.5);
    beta ~ normal(0, 1);
    sigmasq ~ inv_gamma(1.5, 6);
    y ~ normal(X * b, sqrt(sigmasq));
}
";
sm = SampleModel("example", model);

# Design matrix
X = hcat(ones(4), vcat(zeros(3)', diagm(ones(3))))
# Prior
Π = [Normal(1, 0.5); Normal.(zeros(3), 1); InverseGamma(1.5, 6.)]
# Likelihood
ℙy = Normal.([-0.5, 0, 0.5, 1], 5)
# Sample data
x, y = generate_data(ℙy, 100, weights([0.25 for _ = 1:4]))
data = (N = length(y), P = size(X, 2), X = X[x, :]', y = y)
# Stan samples
rc = stan_sample(sm; data = data, num_samples = 5000);
st = read_samples(sm, :dataframe)
# Laplace
μlap, Σlap = LaplaceApproximation(lposterior, X[x, :], y, Π)
μlap[5] = exp(μlap[5]);

# Comparison
hcat(μlap, [mean(c) for c in eachcol(st)[1:5]])
hcat([mean(c .> 0) for c in eachcol(st)[1:5]], 1 .- cdf.(Normal.(μlap, sqrt.(diag(Σlap))), 0))
