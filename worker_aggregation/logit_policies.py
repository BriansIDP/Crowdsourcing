from typing import Union

import numpy as np
import numpy.linalg as nl

def find_kl_gaussians(mu1, var1, mu2, var2):
    mu1 = np.float32(mu1)
    mu2 = np.float32(mu2)
    var1 = np.float32(var1)
    var2 = np.float32(var2)
    return 0.5*(np.log(var2/var1) - 1 + var1/var2 + (mu1-mu2)**2/var2)

class EMGaussian:

    def __init__(self,
            prior_var_of_bias: float,
            prior_mean_of_cov_diag_el: float,
            prior_var_of_cov: float, 
            estimates: np.ndarray,
            max_iters: int=100, 
            tolerance: float=1e-15, 
            bias_known: bool=False,
            prior_var_of_outcomes: float=1,
            precision_init: Union[np.ndarray,None]=None,
            bias_init: Union[np.ndarray,None]=None,) -> None:
        """
        Parameters:
            prior_var_of_bias: float
                variance of the prior on the bias
            prior_mean_of_cov_diag_el: float
                common diagonal element of the mean of the prior on the covariance matrix
            prior_var_of_cov: float
                inverse concentration parameter of the prior on the covariance matrix
            estimates: np.ndarray
                matrix of size timesteps x dim which contains the estimates of the policys
            max_iters: int
                max iterations for which the em algorithm is run
            tolerance: float
                when the average mse for predicitions does not change beyond tolerance, then we stop
            bias_known: bool
                if true, then the bias is not estimated and is set to the initial value
            prior_var_of_outcomes: float
                variance of the prior on the outcomes
            precision_init: np.ndarray
                matrix of size dim x dim which is the initial estimate of the precision matrix 
                for the em algorithm
            bias_init: np.ndarray
                vector of size dim which is the initial estimate of the bias for the em algorithm 
        """
        # TODO: modify the code to handle various prior_var_of_outcomes
        # assert prior_var_of_outcomes==1, "implement code for the case where prior_var_of_outcomes is not 1"
        self._prior_var_of_bias = prior_var_of_bias
        self._estimates = estimates
        self._timesteps, self._dim = self._estimates.shape
        self._prior_mean_of_cov = np.eye(self._dim)*prior_mean_of_cov_diag_el
        self._prior_conc_of_cov = 1/prior_var_of_cov 
        self._max_iters = max_iters
        self._tolerance = tolerance
        self._bias_known = bias_known
        self._prior_var_of_outcomes = prior_var_of_outcomes
        if bias_init is None:
            self._bias_init = np.zeros(self._estimates.shape[1])
        else:
            self._bias_init = bias_init
        if precision_init is None:
            self._precision_init = nl.inv(self.prior_mean_of_cov)
        else:
            self._precision_init = precision_init

    def _do_e_step(self,
                   precision_est:np.ndarray, 
                   bias_est:np.ndarray,) -> None:
        ones_vec = np.ones((self._dim, 1))
        bias_est = bias_est.reshape(-1,1)
        z_var = 1/(1/self._prior_var_of_outcomes + np.sum(precision_est))
        z_mean = (self._estimates-np.ones(self._estimates.shape)*bias_est.T)@(precision_est@ones_vec)*z_var
        return z_mean, z_var

    def _do_approx_m_step(self,
                           z_mean: np.ndarray,
                           z_var: np.ndarray,
                           precision_est: np.ndarray):
            temp_vec = self._estimates - (z_mean*np.ones(self._estimates.shape))
            # bias_est = nl.inv(np.eye(self._dim)/self._prior_var_of_bias + 
            #                 self._timesteps*precision_est)@precision_est@np.sum(temp_vec, axis=0)
            if self._bias_known:
                bias_est = self._bias_init
            else:
                bias_est = nl.solve(np.eye(self._dim)/self._prior_var_of_bias + self._timesteps*precision_est, 
                                    precision_est@np.sum(temp_vec, axis=0))
            
            temp_vec_debiased = temp_vec - bias_est*np.ones(temp_vec.shape)
            temp_mat = temp_vec_debiased.T@temp_vec_debiased/self._timesteps # normalizing
            
            cov_est = (self._prior_conc_of_cov*self._prior_mean_of_cov/self._timesteps \
                    + temp_mat + z_var*np.ones((self._dim, self._dim)))/((self._prior_conc_of_cov\
                                                                        +2*self._dim+2)/self._timesteps + 1)
            precision_est = nl.inv(cov_est)
            return bias_est, precision_est

    def run(self,):
        err_arr = []
        precision_est = self._precision_init
        bias_est = self._bias_init
        for iter in range(self._max_iters):
            if self._estimates.shape[0]==0:
                precision_est = nl.inv(self._prior_mean_of_cov)
                bias_est = self._bias_init
                break
            # E-step
            z_mean, z_var = self._do_e_step(precision_est=precision_est, bias_est=bias_est)

            # early stopping condition
            if iter>1:
                err_arr.append(np.mean((z_mean - z_mean_prev)**2))
                if np.mean((z_mean - z_mean_prev)**2)<self._tolerance:
                    break
            
            # M-step : solve an optimization problem
            out_m_step = self._do_approx_m_step(z_mean=z_mean, z_var=z_var, precision_est=precision_est)

            bias_est, precision_est = out_m_step
            z_mean_prev = z_mean.copy()

        weights = precision_est @ np.ones(self._dim)/(1/self._prior_var_of_outcomes + np.sum(precision_est))
        bias = bias_est
        return {
            'weights': weights,
            'bias': bias,
            'precision_est': precision_est,
            'total_iters': iter+1,
        }