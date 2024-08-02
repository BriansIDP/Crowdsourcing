from typing import Union

import numpy as np
import numpy.linalg as nl
from tqdm import trange
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from .utils import find_kl_gaussians
from .utils import gaussian_log_likelihood
from .utils import TwoLayerMLP, train_neural_net

class EMGaussian:

    def __init__(self,
            num_workers: int,
            prior_var_of_bias: float,
            prior_mean_of_cov_diag_el: float,
            prior_var_of_cov: float, 
            max_iters: int=100, 
            tolerance: float=1e-15, 
            bias_known: bool=False,
            prior_var_of_outcomes: float=1,
            precision_init: Union[np.ndarray,None]=None,
            bias_init: Union[np.ndarray,None]=None,) -> None:
        """
        Parameters:
            num_workers: int
                number of workers giving estimates
            prior_var_of_bias: float
                variance of the prior on the bias
            prior_mean_of_cov_diag_el: float
                common diagonal element of the mean of the prior on the covariance matrix
            prior_var_of_cov: float
                inverse concentration parameter of the prior on the covariance matrix
            max_iters: int
                max iterations for which the em algorithm is run
            tolerance: float
                when the average mse for predicitions does not change beyond tolerance, then we stop
            bias_known: bool
                if true, then the bias is not estimated and is set to the initial value
            prior_var_of_outcomes: float
                variance of the prior on the outcomes
            precision_init: np.ndarray
                matrix of size num_workers x num_workers which is the initial estimate of the precision matrix 
                for the em algorithm
            bias_init: np.ndarray
                vector of size num_workers which is the initial estimate of the bias for the em algorithm 
        """
        self.prior_var_of_bias = prior_var_of_bias
        self.num_workers = num_workers
        self.prior_mean_of_cov = np.eye(self.num_workers)*prior_mean_of_cov_diag_el
        self.prior_conc_of_cov = 1/prior_var_of_cov 
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.bias_known = bias_known
        self.prior_var_of_outcomes = prior_var_of_outcomes
        if bias_init is None:
            self.bias_init = np.zeros(num_workers)
        else:
            self.bias_init = bias_init
        if precision_init is None:
            self.precision_init = nl.inv(self.prior_mean_of_cov)
        else:
            self.precision_init = precision_init

    def do_e_step(self,
                   precision_est:np.ndarray, 
                   bias_est:np.ndarray,) -> None:
        ones_vec = np.ones((self.num_workers, 1))
        bias_est = bias_est.reshape(-1,1)
        z_var = 1/(1/self.prior_var_of_outcomes + np.sum(precision_est))
        z_mean = (self.estimates-np.ones(self.estimates.shape)*bias_est.T)@(precision_est@ones_vec)*z_var
        return z_mean, z_var

    def do_approx_m_step(self,
                           z_mean: np.ndarray,
                           z_var: np.ndarray,
                           precision_est: np.ndarray):
        temp_vec = self.estimates - (z_mean*np.ones(self.estimates.shape))
        # bias_est = nl.inv(np.eye(self.num_workers)/self.prior_var_of_bias + 
        #                 self.timesteps*precision_est)@precision_est@np.sum(temp_vec, axis=0)
        if self.bias_known:
            bias_est = self.bias_init
        else:
            bias_est = nl.solve(np.eye(self.num_workers)/self.prior_var_of_bias + self.timesteps*precision_est, 
                                precision_est@np.sum(temp_vec, axis=0))
        
        temp_vec_debiased = temp_vec - bias_est*np.ones(temp_vec.shape)
        temp_mat = temp_vec_debiased.T@temp_vec_debiased/self.timesteps # normalizing
        
        cov_est = (self.prior_conc_of_cov*self.prior_mean_of_cov/self.timesteps \
                + temp_mat + z_var*np.ones((self.num_workers, self.num_workers)))/((self.prior_conc_of_cov\
                                                                    +2*self.num_workers+2)/self.timesteps + 1)
        precision_est = nl.inv(cov_est)
        return bias_est, precision_est

    def fit(self, estimates: np.ndarray,):
        self.estimates = estimates
        self.timesteps = self.estimates.shape[0]
        err_arr = []
        precision_est = self.precision_init
        bias_est = self.bias_init
        for iter in range(self.max_iters):
            if self.estimates.shape[0]==0:
                precision_est = nl.inv(self.prior_mean_of_cov)
                bias_est = self.bias_init
                break
            # E-step
            z_mean, z_var = self.do_e_step(precision_est=precision_est, bias_est=bias_est)

            # early stopping condition
            if iter>1:
                err_arr.append(np.mean((z_mean - z_mean_prev)**2))
                if np.mean((z_mean - z_mean_prev)**2)<self.tolerance:
                    break
            
            # M-step : solve an optimization problem
            out_m_step = self.do_approx_m_step(z_mean=z_mean, z_var=z_var, precision_est=precision_est)

            bias_est, precision_est = out_m_step
            z_mean_prev = z_mean.copy()

        weights = precision_est @ np.ones(self.num_workers)/(1/self.prior_var_of_outcomes + np.sum(precision_est))
        bias = bias_est
        self.weights = weights
        self.bias = bias
        return {
            'weights': weights,
            'bias': bias,
            'precision_est': precision_est,
            'total_iters': iter+1,
        }
    
    def predict(self, estimates: np.ndarray) -> np.ndarray:
        group_ests = (estimates - self.bias)@self.weights
        preds = np.array(group_ests > 0, dtype=np.int32)
        return preds
            
class EM_GMM:

    def __init__(self,
                 num_workers: int,
                 cov_mat_diag: float,
                 mean0_el: float=-1,
                 mean1_el: float=1,
                 max_iter: int=100,
                 tol: float=1e-8) -> None:
        self.cov0 = np.eye(num_workers)*cov_mat_diag
        self.cov1 = np.eye(num_workers)*cov_mat_diag
        self.mean0 = mean0_el*np.ones(num_workers)
        self.mean1 = mean1_el*np.ones(num_workers)
        self.max_iter = max_iter
        self.tol = tol
    
    def do_e_step(self, ests: np.ndarray):
        loglik0 = gaussian_log_likelihood(ests, mean=self.mean0, cov=self.cov0)
        loglik1 = gaussian_log_likelihood(ests, mean=self.mean1, cov=self.cov1)
        log_prob1 = loglik1 - np.logaddexp(loglik0, loglik1)
        prob1 = np.exp(log_prob1)
        assert np.all(prob1 >= 0) and np.all(prob1 <= 1)
        return prob1

    def do_m_step(self, ests: np.ndarray, prob1: np.ndarray):
        N = len(ests)
        prob0 = 1-prob1
        assert np.all(prob0 >= 0) and np.all(prob0 <= 1)
        mean0 = np.mean(ests*prob0[:, None], axis=0)/np.mean(prob0)
        mean1 = np.mean(ests*prob1[:, None], axis=0)/np.mean(prob1)
        temp0 = (ests - mean0[None, :])*prob0[:, None]**0.5/N**0.5
        temp1 = (ests - mean1[None, :])*prob1[:, None]**0.5/N**0.5
        # for i in range(N):
        #     cov0 += prob0[i]*np.outer(ests[i]-mean0, ests[i]-mean0)
        #     cov1 += prob1[i]*np.outer(ests[i]-mean1, ests[i]-mean1)
        cov0 = temp0.T@temp0
        cov1 = temp1.T@temp1
        assert cov0.shape == (len(mean0), len(mean0))
        assert cov1.shape == (len(mean1), len(mean1))
        assert np.allclose(cov0, cov0.T)
        assert np.allclose(cov1, cov1.T)
        cov0 /= np.mean(prob0)
        cov1 /= np.mean(prob1)
        return mean0, mean1, cov0, cov1
    
    def fit(self, ests: np.ndarray,) -> None:
        for i in trange(self.max_iter):
            prob1 = self.do_e_step(ests)
            mean0, mean1, cov0, cov1 = self.do_m_step(ests, prob1)
            if i>0:
                # if np.max(np.abs(mean0-mean0_prev))<self.tol:
                #     break
                if nl.norm(prob1_prev-prob1)/len(prob1)**0.5<self.tol:
                    # print(nl.norm(prob1_prev-prob1)/len(prob1)**0.5)
                    break
            # mean0_prev = mean0.copy()
            prob1_prev = prob1.copy()
            self.mean0 = mean0
            self.mean1 = mean1
            self.cov0 = cov0
            self.cov1 = cov1
    
    def predict(self, ests: np.ndarray) -> np.ndarray:
        prob1 = self.do_e_step(ests)
        return np.array(prob1 > 0.5, dtype=np.int32)

class AvgSSLPreds:
    def __init__(self, 
                 neural_nets: nn.Module,
                 num_workers: int,
                 loss_fn_type: str="bce_logit",
                 lr: float=1e-3, 
                 weight_decay: float=1e-5,
                 patience: int=20,
                 epochs: int=1000,
                 use_joblib_fit: bool=False,
                 use_joblib_seeds: bool=True,
                 ) -> None:
        self.neural_nets = neural_nets
        self.num_workers = num_workers
        assert len(self.neural_nets) == self.num_workers
        self.loss_fn_type = loss_fn_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.epochs = epochs
        if use_joblib_seeds:
            assert not use_joblib_fit
        self.use_joblib_fit = use_joblib_fit
    
    def fit(self, estimates: np.ndarray, testing=False) -> None:
        # self.scaler = StandardScaler()
        # estimates = self.scaler.fit_transform(estimates)
        sigmoid = lambda x: 1/(1+np.exp(-x))
        estimates = sigmoid(estimates)
        estimates_train = estimates[:int(0.8*estimates.shape[0])]
        estimates_val = estimates[int(0.8*estimates.shape[0]):]
        results = []

        def create_data_dict(idx, data, val_data):
            other_ids = [j for j in range(estimates.shape[1]) if j != idx]
            x_train = torch.tensor(data[:, other_ids], dtype=torch.float32)
            x_val = torch.tensor(val_data[:, other_ids], dtype=torch.float32)
            y_train = torch.tensor(data[:, idx], dtype=torch.float32).reshape(-1, 1)
            y_val = torch.tensor(val_data[:, idx], dtype=torch.float32).reshape(-1, 1)
            return {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val}
            
        if not self.use_joblib_fit:
            for i in trange(estimates.shape[1]):
                result = train_neural_net(**create_data_dict(i, estimates_train, estimates_val),
                                          neural_net=self.neural_nets[i], loss_fn_type=self.loss_fn_type,
                                          lr=self.lr, weight_decay=self.weight_decay, patience=self.patience,
                                          epochs=self.epochs, testing=testing)
                results.append(result)
        else:
            results = Parallel(n_jobs=-1)(
                            delayed(train_neural_net)(
                                **create_data_dict(i, estimates_train, estimates_val),
                                neural_net=self.neural_nets[i],
                                loss_fn_type=self.loss_fn_type,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                patience=self.patience,
                                epochs=self.epochs,
                                testing=testing
                            ) for i in range(self.num_workers)
                        )
        self.models = []
        self.stats_dict = {}
        for i, (model, stats_dict_i) in enumerate(results):
            self.models.append(model)
            if i == 0:
                for key in stats_dict_i:
                    self.stats_dict[key] = [stats_dict_i[key]]
            else:
                for key in stats_dict_i:
                    self.stats_dict[key].append(stats_dict_i[key])
    
    def predict(self, estimates: np.ndarray, testing=False) -> np.ndarray:
        # estimates = self.scaler.transform(estimates)
        sigmoid = lambda x: 1/(1+np.exp(-x))
        estimates = sigmoid(estimates)
        estimates_tensor = torch.tensor(estimates, dtype=torch.float32)
        preds = np.zeros(estimates.shape)*np.nan
        for i in range(estimates.shape[1]):
            self.models[i].eval()
            x_test = estimates_tensor[:, [j for j in range(estimates.shape[1]) if j != i]]
            if self.loss_fn_type == "bce_logit":
                preds[:,i] = torch.sigmoid(self.models[i](x_test)).detach().numpy().flatten()
            elif self.loss_fn_type == "mse":
                preds[:,i] = self.models[i](x_test).detach().numpy().flatten()
            else:
                raise ValueError(f"loss_fn_type={self.loss_fn_type} not recognized")
        # preds = self.scaler.inverse_transform(preds)
        group_ests = np.mean(preds, axis=1)
        labels = np.array(group_ests > 0.5, dtype=np.int32)
        if testing:
            return labels, group_ests, preds
        return labels
