import numpy as np

def find_kl_gaussians(mu1, var1, mu2, var2):
    mu1 = np.float32(mu1)
    mu2 = np.float32(mu2)
    var1 = np.float32(var1)
    var2 = np.float32(var2)
    return 0.5*(np.log(var2/var1) - 1 + var1/var2 + (mu1-mu2)**2/var2)

def gaussian_log_likelihood(x: np.ndarray, 
                            mean: np.ndarray, 
                            cov: np.ndarray) -> float:
    N, dim = x.shape
    assert mean.shape == (dim,)
    assert cov.shape == (dim, dim)
    
    # Calculate log determinant of the covariance matrix
    sign, logdet = np.linalg.slogdet(cov)
    if sign != 1:
        raise ValueError("Covariance matrix must be positive definite.")
    
    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)
    
    # Compute the log likelihood
    diff = x - mean[None, :]
    # log_likelihood = -0.5 * (logdet + diff.T @ inv_cov @ diff + len(x) * np.log(2 * np.pi))
    log_likelihood = -0.5*logdet*np.ones(N) - 0.5*np.sum(diff @ inv_cov * diff, axis=1) \
        - 0.5*dim*np.log(2*np.pi)
    assert log_likelihood.shape == (N,)
    
    return log_likelihood

