from typing import Union

import pytest
import numpy as np
import numpy.linalg as nl

from worker_aggregation import EMGaussian

def mse_based_on_weights(cov_true: np.ndarray, bias_true: np.ndarray, 
                         prior_var_of_outcomes: float,
                         weights: np.ndarray, bias: Union[np.ndarray, float]):
    dim = cov_true.shape[0]
    opt_weights = nl.solve(cov_true + prior_var_of_outcomes*np.ones((dim, dim)), 
                           prior_var_of_outcomes*np.ones(dim))
    # following line uses two facts
    # loss = v + w^T (Sigma + v 1 1^T) w - 2 v w^T 1
    # (Sigma + v 1 1^T) w_* = v 1
    aleatoric_mse = prior_var_of_outcomes*(1 - np.sum(opt_weights))
    # print(f"aleatoric mse: {aleatoric_mse}")
    diff = weights - opt_weights
    epistemic_mse_weights = diff@(cov_true + prior_var_of_outcomes*np.ones((dim, dim)))@diff.T
    if isinstance(bias, float):
        epistemic_mse_bias = (-bias - bias_true@weights)**2
    elif isinstance(bias, np.ndarray):
        epistemic_mse_bias = np.dot((bias - bias_true), weights)**2
    else:
        raise ValueError("bias should be either float or numpy array")
    mse = aleatoric_mse + epistemic_mse_weights + epistemic_mse_bias 
    return mse


def generate_data(dim, bias, data_seed=0, timesteps=2000, prior_mean=0, prior_var=1):
    cov_mat = nl.inv(np.array(
            [[1/5, 1/10, 0],
             [1/10, 1/5, 1/10],
             [0, 1/10, 1/5]]
        ))
    rng = np.random.default_rng(seed=data_seed)
    outcomes = rng.normal(prior_mean, prior_var, size=timesteps)
    estimates = np.array([rng.multivariate_normal(mean=outcome*np.ones(dim)+bias,
                                                  cov=cov_mat) for outcome in outcomes])
    return {"cov_mat": cov_mat, "outcomes": outcomes, "estimates": estimates, "prior_var": prior_var, 
            "bias": bias}

@pytest.fixture
def zero_bias():
    dim = 3
    bias = np.zeros(dim)
    return generate_data(dim, bias)

@pytest.fixture
def non_zero_bias():
    dim = 3
    bias = np.array([1/2**0.5, 0, -1/2**0.5])
    return generate_data(dim, bias)

@pytest.fixture
def small_prior_var():
    dim = 3
    bias = np.zeros(dim)
    return generate_data(dim, bias, prior_var=0.1)

class TestPolicy:

    def test_zero_bias(self, zero_bias):
        estimates = zero_bias["estimates"]
        timesteps, dim = estimates.shape
        prior_var_of_bias = 10
        prior_mean_of_cov_diag_el = 10
        prior_var_of_cov = 1
        policy_seed = 1
        rng = np.random.default_rng(seed=policy_seed)
        precision_init = np.diag([rng.gamma(shape=10)]*dim)
        bias_init = rng.multivariate_normal(mean=np.zeros(dim), 
                                            cov=prior_var_of_bias*np.eye(dim))
        alg = EMGaussian(estimates=estimates,
                         prior_var_of_bias=prior_var_of_bias, prior_mean_of_cov_diag_el=prior_mean_of_cov_diag_el, 
                         prior_var_of_cov=prior_var_of_cov, precision_init=precision_init,
                         bias_init=bias_init)
        
        out = alg.run()
        weights = out["weights"]
        bias = out["bias"]

        prior_var = zero_bias["prior_var"]
        cov_true = zero_bias["cov_mat"]
        optimal_weights = nl.solve(cov_true + prior_var*np.ones((dim, dim)), prior_var*np.ones(dim))
        bias_true = zero_bias["bias"]
        var_true = 1/(1/prior_var + np.sum(nl.inv(cov_true)))

        mse = mse_based_on_weights(cov_true, bias_true, prior_var, weights, bias)

        assert(np.isclose(mse, var_true, atol=1e-2))
        print(f"optimal weights: {optimal_weights}")
        print(f"weights: {weights}")
        print(f"Bias true: {bias_true}")
        print(f"Bias est: {bias}")

    def test_non_zero_bias(self, non_zero_bias):
        estimates = non_zero_bias["estimates"]
        timesteps, dim = estimates.shape
        prior_var_of_bias = 10
        prior_mean_of_cov_diag_el = 10
        prior_var_of_cov = 1
        policy_seed = 1
        rng = np.random.default_rng(seed=policy_seed)
        precision_init = np.diag([rng.gamma(shape=10)]*dim)
        bias_init = rng.multivariate_normal(mean=np.zeros(dim), 
                                            cov=prior_var_of_bias*np.eye(dim))
        alg = EMGaussian(estimates=estimates,
                         prior_var_of_bias=prior_var_of_bias, 
                         prior_mean_of_cov_diag_el=prior_mean_of_cov_diag_el, 
                         prior_var_of_cov=prior_var_of_cov, precision_init=precision_init,
                         bias_init=bias_init)
        
        out = alg.run()
        weights = out["weights"]
        bias = out["bias"]

        prior_var = non_zero_bias["prior_var"]
        cov_true = non_zero_bias["cov_mat"]
        optimal_weights = nl.solve(cov_true + prior_var*np.ones((dim, dim)), prior_var*np.ones(dim))
        bias_true = non_zero_bias["bias"]
        var_true = 1/(1/prior_var + np.sum(nl.inv(cov_true)))

        mse = mse_based_on_weights(cov_true, bias_true, prior_var, weights, bias)

        assert(np.isclose(mse, var_true, atol=1e-2))
        print(f"optimal weights: {optimal_weights}")
        print(f"weights: {weights}")
        print(f"Bias true: {bias_true}")
        print(f"Bias est: {bias}")
    
    def test_small_prior_var(self, small_prior_var):
        estimates = small_prior_var["estimates"]
        timesteps, dim = estimates.shape
        prior_var_of_bias = 10
        prior_mean_of_cov_diag_el = 10
        prior_var_of_cov = 1
        policy_seed = 1
        rng = np.random.default_rng(seed=policy_seed)
        precision_init = np.diag([rng.gamma(shape=10)]*dim)
        bias_init = rng.multivariate_normal(mean=np.zeros(dim), 
                                            cov=prior_var_of_bias*np.eye(dim))
        prior_var = small_prior_var["prior_var"]
        alg = EMGaussian(estimates=estimates,
                         prior_var_of_bias=prior_var_of_bias, 
                         prior_mean_of_cov_diag_el=prior_mean_of_cov_diag_el, 
                         prior_var_of_cov=prior_var_of_cov, precision_init=precision_init,
                         bias_init=bias_init, prior_var_of_outcomes=prior_var)
        
        out = alg.run()
        weights = out["weights"]
        bias = out["bias"]

        cov_true = small_prior_var["cov_mat"]
        optimal_weights = nl.solve(cov_true + prior_var*np.ones((dim, dim)), prior_var*np.ones(dim))
        bias_true = small_prior_var["bias"]
        var_true = 1/(1/prior_var + np.sum(nl.inv(cov_true)))

        mse = mse_based_on_weights(cov_true, bias_true, prior_var, weights, bias)

        assert(np.isclose(mse, var_true, atol=1e-2))
        print(f"optimal weights: {optimal_weights}")
        print(f"weights: {weights}")
        print(f"Bias true: {bias_true}")
        print(f"Bias est: {bias}")
