from typing import Union

import numpy as np
import numpy.linalg as nl
from tqdm import trange
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .utils import find_kl_gaussians
from .utils import gaussian_log_likelihood
from .utils import TwoLayerMLP, train_neural_net
from .utils import CustomDataset, TrainWithLoaders

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
                 neural_net_cons,
                 num_workers: int,
                 loss_fn_type: str="bce_logit",
                 lr: float=1e-3, 
                 weight_decay: float=1e-5,
                 patience: int=20,
                 epochs: int=1000,
                 use_joblib_fit: bool=False,
                 use_joblib_multirun: bool=True,
                 logits: bool=True,
                 folds: int=5,
                 fold: Union[int, None]=None
                 ) -> None:
        print(f"lr: {lr}, weight_decay: {weight_decay}, patience: {patience}, epochs: {epochs}")
        # self.neural_nets = neural_nets
        self.neural_net_cons = neural_net_cons
        self.num_workers = num_workers
        # assert len(self.neural_nets) == self.num_workers
        self.loss_fn_type = loss_fn_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.epochs = epochs
        if use_joblib_multirun:
            assert not use_joblib_fit
        self.use_joblib_fit = use_joblib_fit
        self.logits = logits
        self.folds = folds
        self.fold = fold
    
    def fit(self, estimates: np.ndarray, testing=False) -> None:
        # self.scaler = StandardScaler()
        # estimates = self.scaler.fit_transform(estimates)
        if self.logits:
            sigmoid = lambda x: 1/(1+np.exp(-x))
            estimates = sigmoid(estimates)
        self.fold_models = []
        for fold in range(self.folds):
            if self.fold is not None and fold != self.fold:
                self.fold_models.append([None]*self.num_workers)
                continue
            print(f"fold: {fold}")
            len_ests = estimates.shape[0]
            val_idx = np.arange(int(fold*len_ests/self.folds), int((fold+1)*len_ests/self.folds))
            train_idx = np.array([i for i in range(len_ests) if i not in val_idx])
            estimates_train = estimates[train_idx]
            estimates_val = estimates[val_idx]
            # estimates_train = estimates[:int(0.8*estimates.shape[0])]
            # estimates_val = estimates[int(0.8*estimates.shape[0]):]

            results = []

            def create_data_dict(idx, data, val_data):
                other_ids = [j for j in range(estimates.shape[1]) if j != idx]
                x_train = torch.tensor(data[:, other_ids], dtype=torch.float32)
                x_val = torch.tensor(val_data[:, other_ids], dtype=torch.float32)
                y_train = torch.tensor(data[:, idx], dtype=torch.float32).reshape(-1, 1)
                y_val = torch.tensor(val_data[:, idx], dtype=torch.float32).reshape(-1, 1)
                return {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val}
                
            if not self.use_joblib_fit:
                for i in trange(self.num_workers):
                    result = train_neural_net(**create_data_dict(i, estimates_train, estimates_val),
                                            neural_net=self.neural_net_cons(), loss_fn_type=self.loss_fn_type,
                                            lr=self.lr, weight_decay=self.weight_decay, patience=self.patience,
                                            epochs=self.epochs, testing=testing)
                    results.append(result)
            else:
                results = Parallel(n_jobs=self.num_workers)(
                                delayed(train_neural_net)(
                                    **create_data_dict(i, estimates_train, estimates_val),
                                    neural_net=self.neural_net_cons(),
                                    loss_fn_type=self.loss_fn_type,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    patience=self.patience,
                                    epochs=self.epochs,
                                    testing=testing
                                ) for i in range(self.num_workers)
                            )
            models = []
            # self.stats_dict = {}
            for i, (model, stats_dict_i) in enumerate(results):
                models.append(model)
                # if i == 0:
                #     for key in stats_dict_i:
                #         self.stats_dict[key] = [stats_dict_i[key]]
                # else:
                #     for key in stats_dict_i:
                #         self.stats_dict[key].append(stats_dict_i[key])
            self.fold_models.append(models)
    
    def predict(self, estimates: np.ndarray, testing=False,) -> np.ndarray:
        # estimates = self.scaler.transform(estimates)
        if self.logits:
            sigmoid = lambda x: 1/(1+np.exp(-x))
            estimates = sigmoid(estimates)
        # estimates_tensor = torch.tensor(estimates, dtype=torch.float32)
        preds = np.zeros(estimates.shape)*np.nan

        if self.fold is None:
            for fold in range(self.folds):
                print(f"fold: {fold}")
                len_ests = estimates.shape[0]
                val_idx = np.arange(int(fold*len_ests/self.folds), int((fold+1)*len_ests/self.folds))
                estimates_val = estimates[val_idx]
                print("estimates_val.shape", estimates_val.shape)
                for i in range(estimates.shape[1]):
                    self.fold_models[fold][i].eval()
                    estimates_tensor = torch.tensor(estimates_val, dtype=torch.float32)
                    x_test = estimates_tensor[:, [j for j in range(estimates.shape[1]) if j != i]]
                    if self.loss_fn_type == "bce":
                        preds[val_idx,i] = \
                            torch.sigmoid(self.fold_models[fold][i](x_test)).detach().numpy().flatten()
                    elif self.loss_fn_type == "mse":
                        preds[val_idx,i] = self.fold_models[fold][i](x_test).detach().numpy().flatten()
                    else:
                        raise ValueError(f"loss_fn_type={self.loss_fn_type} not recognized")
            # preds = self.scaler.inverse_transform(preds)
        else:
            fold = self.fold
            print(f"fold: {fold}")
            for i in range(estimates.shape[1]):
                self.fold_models[fold][i].eval()
                estimates_tensor = torch.tensor(estimates, dtype=torch.float32)
                x_test = estimates_tensor[:, [j for j in range(estimates.shape[1]) if j != i]]
                if self.loss_fn_type == "bce":
                    preds[:,i] = \
                        torch.sigmoid(self.fold_models[fold][i](x_test)).detach().numpy().flatten()
                elif self.loss_fn_type == "mse":
                    preds[:,i] = self.fold_models[fold][i](x_test).detach().numpy().flatten()
                else:
                    raise ValueError(f"loss_fn_type={self.loss_fn_type} not recognized")
        group_ests = np.mean(preds, axis=1)
        labels = np.array(group_ests > 0.5, dtype=np.int32)
        if testing:
            return labels, group_ests, preds
        return labels, group_ests

class Averaging:
    def __init__(self,
            num_workers: int,
            apply_sigmoid: bool=False,
            logits: bool=True) -> None:
        self.num_workers = num_workers
        self.apply_sigmoid = apply_sigmoid
        # print(f"apply_sigmoid: {apply_sigmoid}")
        self.logits = logits
        if not self.logits:
            assert not self.apply_sigmoid, "apply_sigmoid should be False when logits is False"
    
    def fit(self, estimates: np.ndarray) -> None:
        pass

    def predict(self, estimates: np.ndarray) -> np.ndarray:
        if self.apply_sigmoid:
            sigmoid = lambda x: 1/(1+np.exp(-x))
            estimates = sigmoid(estimates)
        group_ests = np.mean(estimates, axis=1)
        if self.logits:
            if self.apply_sigmoid:
                labels = np.array(group_ests > 0.5, dtype=np.int32)
            else:
                labels = np.array(group_ests > 0, dtype=np.int32)
        else:
            labels = np.array(group_ests > 0.5, dtype=np.int32)
        return labels, group_ests

class AvgSSLPredsContextVec:
    def __init__(self, 
                 neural_net_cons,
                 num_workers: int,
                 seed: int,
                 batch_size: int=100,
                 loss_fn_type: str="bce",
                 lr: float=1e-3, 
                 weight_decay: float=1e-5,
                 patience: int=2,
                 max_grad_steps: int=1000,
                 use_joblib_fit: bool=False,
                 use_joblib_multirun: bool=True,
                 eval_interval: int=100,
                 lr_scheduler_type: str='constant',
                 folds: int=5) -> None:
        self.neural_net_cons = neural_net_cons
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.loss_fn_type = loss_fn_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_grad_steps = max_grad_steps
        if use_joblib_multirun:
            assert not use_joblib_fit
        self.use_joblib_multirun = use_joblib_multirun
        self.use_joblib_fit = use_joblib_fit
        self.eval_interval = eval_interval
        self.lr_scheduler_type = lr_scheduler_type
        self.folds = folds
    
    def create_collate_fn(self, i: int):
        def collate_fn(batch):
            contexts, ests = zip(*batch)
            contexts = torch.stack(contexts)
            other_ids = [j for j in range(self.num_workers) if j!=i]
            ests = torch.stack(ests)
            other_ests = ests[:, other_ids].float()
            x = torch.cat((contexts, other_ests), dim=1)
            # x = torch.cat((torch.tensor(contexts), other_ests), dim=1)
            if self.loss_fn_type == 'bce':
                y = ests[:, i:i+1].float()
            elif self.loss_fn_type == 'mse':
                y = ests[:, i:i+1].float()
            else:
                raise ValueError("loss_fn_type should be 'bce' or 'mse'")
            return x, y
        return collate_fn
    
    def fit(self, ests: np.ndarray, contexts: np.ndarray) -> None:
        def fit_i(i):
            train_dataloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=False, # data is already shuffled because of perm below
                collate_fn=self.create_collate_fn(i),
                # generator=rng,
            )
            val_dataloader = DataLoader(
                val_data,
                batch_size=len(val_data),
                shuffle=False,
                collate_fn=self.create_collate_fn(i),
            )
            result = TrainWithLoaders(
                        neural_net=self.neural_net_cons(),
                        train_loader=train_dataloader, 
                        val_loader=val_dataloader,
                        lr=self.lr, 
                        weight_decay=self.weight_decay, 
                        patience=self.patience,
                        max_grad_steps=self.max_grad_steps,
                        loss_fn_type=self.loss_fn_type,
                        lr_scheduler_type=self.lr_scheduler_type,
                        eval_interval=self.eval_interval,
                        use_joblib_fit=self.use_joblib_fit,
                        use_joblib_multirun=self.use_joblib_multirun,).run()
            return result

        # rng = torch.Generator().manual_seed(self.seed)
        self.fold_models = []
        for fold in range(self.folds):
            print(f"fold: {fold}")
            len_ests = ests.shape[0]
            val_idx = np.arange(int(fold*len_ests/self.folds), int((fold+1)*len_ests/self.folds))
            train_idx = np.array([i for i in range(len_ests) if i not in val_idx])
            ests_train = ests[train_idx]
            ests_val = ests[val_idx]
            contexts_train = contexts[train_idx]
            contexts_val = contexts[val_idx]
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(ests_train.shape[0])
            train_data = CustomDataset(contexts_train[perm], ests_train[perm])
            val_data = CustomDataset(contexts_val, ests_val)
            results = []
            if not self.use_joblib_fit:
                for i in range(self.num_workers):
                    print(f"Training model {i}")
                    results.append(fit_i(i))
            else:
                results = Parallel(n_jobs=ests.shape[1])(
                                delayed(fit_i)(i) for i in range(self.num_workers)
                            )
            models = []
            for model, _ in results:
                models.append(model)
            self.fold_models.append(models)

    def predict(self, ests: np.ndarray, 
                contexts: np.ndarray, testing: bool=False):
        ssl_preds = []
        group_ests = []
        for fold in range(self.folds):
            len_ests = ests.shape[0]
            val_idx = np.arange(int(fold*len_ests/self.folds), int((fold+1)*len_ests/self.folds))
            ests_val = torch.tensor(ests[val_idx])
            contexts_val = torch.tensor(contexts[val_idx])
            ssl_preds_ = []
            for i in range(self.num_workers):
                self.fold_models[fold][i].eval()
                other_ids = [j for j in range(self.num_workers) if j!=i]
                other_ests = ests_val[:, other_ids].float()
                x_val = torch.cat((contexts_val, other_ests), dim=1)
                if self.loss_fn_type == 'bce':
                    ssl_preds_.append(torch.sigmoid(self.fold_models[fold][i](x_val)).view(-1))
                elif self.loss_fn_type == 'mse':
                    ssl_preds_.append(self.fold_models[fold][i](x_val).view(-1))
                else:
                    raise ValueError("loss_fn_type should be 'bce' or 'mse'")
            ssl_preds_ = torch.stack(ssl_preds_).transpose(1, 0).detach().numpy()
            assert ssl_preds_.shape == (len(val_idx), self.num_workers)
            ssl_preds = np.concatenate((ssl_preds, ssl_preds_)) if len(ssl_preds)>0 else ssl_preds_
            group_ests_ = np.mean(ssl_preds_, axis=1)
            assert group_ests_.shape == (len(val_idx),)
            group_ests = np.concatenate((group_ests, group_ests_)) if len(group_ests)>0 else group_ests_
        probs = group_ests.copy()
        group_ests = (group_ests>0.5).astype(np.int32)
        if not testing:
            return group_ests, probs
        else:
            return group_ests, probs, ssl_preds
