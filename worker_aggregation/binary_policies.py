from typing import Union

import numpy as np
from tqdm import trange
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import train_neural_net
from sklearn.preprocessing import StandardScaler

class EMSymmetricBinary:
    
    def __init__(self, seed: int, 
                 num_workers: int,
                 skill_init: np.ndarray = None,
                 tol: float = 1e-8, max_iter: int = 100):
        self.rng = np.random.default_rng(seed)
        self.num_workers = num_workers
        # self.pi = np.ones(num_classes) / num_classes
        # skill: probability of answering correctly
        if skill_init is not None:
            self.skill = skill_init
        else:
            self.skill = (self.rng.random(num_workers)+1)/2
        assert np.all(self.skill >= 0.5) and np.all(self.skill <= 1)
        # assert self.theta.shape == (2, num_workers)
        self.epsilon = tol
        self.max_iter = max_iter
    
    def fit(self, ests: np.ndarray):
        num_samples = ests.shape[0]
        assert ests.shape[1] == self.num_workers
        # probability that the true label is 1
        prob_1 = np.zeros(num_samples)*np.nan
        for i in trange(self.max_iter):
            # E-step
            prob_1 = self.e_step(ests)
            # M-step
            self.skill = self.m_step(ests, prob_1)
            # stopping criterion on skill change
            if i>0:
                skill_change = self.skill - old_skill
                if np.max(np.abs(skill_change)) < self.epsilon:
                    break
            old_skill = self.skill.copy()
        return
    
    def e_step(self, ests: np.ndarray):
        num_samples = ests.shape[0]
        prob_1 = np.zeros(num_samples)*np.nan
        log_unnorm_probs = np.ones((num_samples, 2))*np.nan
        log_unnorm_probs[:, 1] = ests@np.log(self.skill)  + (1-ests)@np.log(1-self.skill)
        log_unnorm_probs[:, 0] = (1-ests)@np.log(self.skill) + ests@np.log(1-self.skill)
        log_prob_1 = log_unnorm_probs[:, 1] - np.logaddexp(log_unnorm_probs[:, 0], log_unnorm_probs[:, 1])
        prob_1 = np.exp(log_prob_1)
        # for j in range(num_samples):
        #     # log_prob_1 = log_unnorm_probs[j, 1] - np.logaddexp(log_unnorm_probs[j, 0], log_unnorm_probs[j, 1])
        #     prob_1[j] = np.exp(log_prob_1[0])
        try:
            assert np.all(prob_1 >= 0) and np.all(prob_1 <= 1)
        except:
            print("Error")
            idx = np.where((prob_1 < 0) | (prob_1 > 1))[0]
            print(idx)
            print(prob_1[idx])
            raise
        return prob_1
    
    def m_step(self, ests: np.ndarray, prob_1: np.ndarray):
        skill = np.zeros(self.num_workers)*np.nan
        for j in range(self.num_workers):
            arr = prob_1*ests[:, j] + (1 - prob_1)*(1 - ests[:, j])
            skill[j] = np.mean(arr)
        try:
            assert np.all(skill >= 0) and np.all(skill <= 1)
        except:
            print("Error")
            print(skill)
            raise
        return skill
    
    def predict(self, ests: np.ndarray):
        prob_1 = self.e_step(ests)
        group_ests = np.array(prob_1 > 0.5, dtype=np.int32)
        return group_ests

class EMAsymmetricBinary:
    
    def __init__(self, seed: int, 
                 num_workers: int,
                 skill_init: Union[np.ndarray, None] = None,
                 tol: float = 1e-8, max_iter: int = 100,
                 reg_m_step: float = 0):
        self.rng = np.random.default_rng(seed)
        self.num_workers = num_workers
        # skill: probability of answering correctly
        if skill_init is not None:
            self.skill = skill_init
        else:
            self.skill = (self.rng.random((num_workers,2))+1)/2
        assert np.all(self.skill >= 0.5) and np.all(self.skill <= 1)
        self.epsilon = tol
        self.max_iter = max_iter
        self.reg_m_step = reg_m_step
    
    def fit(self, ests: np.ndarray):
        num_samples = ests.shape[0]
        assert ests.shape[1] == self.num_workers
        # probability that the true label is 1
        prob_1 = np.zeros(num_samples)*np.nan
        for i in trange(self.max_iter):
            # E-step
            prob_1 = self.e_step(ests)
            # M-step
            self.skill = self.m_step(ests, prob_1)
            # stopping criterion on skill change
            if i>0:
                skill_change = self.skill - old_skill
                if np.max(np.abs(skill_change)) < self.epsilon:
                    break
            old_skill = self.skill.copy()
        return
    
    def e_step(self, ests: np.ndarray):
        num_samples = ests.shape[0]
        prob_1 = np.zeros(num_samples)*np.nan
        log_unnorm_probs = np.ones((num_samples, 2))*np.nan
        log_unnorm_probs[:, 1] = ests@np.log(self.skill[:,1]) \
            + (1-ests)@np.log(1-self.skill[:,1])
        log_unnorm_probs[:, 0] = (1-ests)@np.log(self.skill[:,0]) \
            + ests@np.log(1-self.skill[:,0])
        log_prob_1 = log_unnorm_probs[:, 1] - np.logaddexp(log_unnorm_probs[:, 0], log_unnorm_probs[:, 1])
        prob_1 = np.exp(log_prob_1)
        assert prob_1.shape == (num_samples,)
        try:
            assert np.all(prob_1 >= 0) and np.all(prob_1 <= 1)
        except:
            print("Error")
            idx = np.where((prob_1 < 0) | (prob_1 > 1))[0]
            print(idx)
            print(prob_1[idx])
            raise
        return prob_1
    
    def m_step(self, ests: np.ndarray, prob_1: np.ndarray):
        skill = np.zeros((self.num_workers,2))*np.nan
        for j in range(self.num_workers):
            ests_j_with_reg = np.append(ests[:, j], 0.5)
            prob_1_with_reg = np.append(prob_1, self.reg_m_step)
            # arr = prob_1*ests[:, j] 
            arr = prob_1_with_reg*ests_j_with_reg
            skill[j,1] = np.mean(arr)/np.mean(prob_1_with_reg)
            one_minus_prob_1_with_reg = np.append(1-prob_1, self.reg_m_step)
            arr = one_minus_prob_1_with_reg*(1-ests_j_with_reg)
            skill[j,0] = np.mean(arr)/np.mean(one_minus_prob_1_with_reg)
        try:
            assert np.all(skill >= 0) and np.all(skill <= 1)
        except:
            print("Error")
            print(skill)
            raise
        return skill
    
    def predict(self, ests: np.ndarray):
        prob_1 = self.e_step(ests)
        group_ests = np.array(prob_1 > 0.5, dtype=np.int32)
        return group_ests

class MajorityVote:
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
    
    def fit(self, ests: np.ndarray):
        return
    
    def predict(self, ests: np.ndarray):
        assert ests.shape[1] == self.num_workers
        group_ests = np.array(np.mean(ests, axis=1) > 0.5, dtype=np.int32)
        return group_ests


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class EMLogisticBinary:
    def __init__(self, seed: int, 
                 context_len: int, num_workers: int,
                 max_iter: int = 100, tol: float = 1e-6,):
        self.rng = np.random.default_rng(seed)
        self.context_len = context_len
        self.num_workers = num_workers
        self.theta = np.random.randn(context_len)
        # skill of each worker is more than 0.5 and is different across classes
        self.skill = (self.rng.random((num_workers,2))+1)/2
        self.max_iter = max_iter
        self.tol = tol
    
    def e_step(self, ests: np.ndarray, contexts: np.ndarray):
        num_samples = ests.shape[0]
        prob_1 = np.zeros(num_samples)*np.nan
        log_unnorm_probs = np.ones((num_samples, 2))*np.nan
        log_unnorm_probs[:, 1] = ests@np.log(self.skill[:,1]) \
            + (1-ests)@np.log(1-self.skill[:,1]) + contexts@self.theta
        log_unnorm_probs[:, 0] = (1-ests)@np.log(self.skill[:,0]) \
            + ests@np.log(1-self.skill[:,0])
        log_prob_1 = log_unnorm_probs[:, 1] \
            - np.logaddexp(log_unnorm_probs[:, 0], log_unnorm_probs[:, 1])
        prob_1 = np.exp(log_prob_1)
        assert prob_1.shape == (num_samples,)
        try:
            assert np.all(prob_1 >= 0) and np.all(prob_1 <= 1)
        except:
            print("Error")
            idx = np.where((prob_1 < 0) | (prob_1 > 1))[0]
            print(idx)
            print(prob_1[idx])
            raise
        return prob_1
    
    def m_step(self, ests: np.ndarray, contexts: np.ndarray, 
               prob_1: np.ndarray):
        skill = np.zeros((self.num_workers,2))*np.nan
        for j in range(self.num_workers):
            arr = prob_1*ests[:, j] 
            skill[j,1] = np.mean(arr)/np.mean(prob_1)
            arr = (1-prob_1)*(1-ests[:, j])
            skill[j,0] = np.mean(arr)/np.mean(1-prob_1)
        try:
            assert np.all(skill >= 0) and np.all(skill <= 1)
        except:
            print("Error")
            print(skill)
            raise

        # Update theta_hat
        n_samples = contexts.shape[0]
        def objective(theta):
            return -np.sum([prob_1[t] * np.log(sigmoid(np.dot(theta, contexts[t]))) + 
                            (1 - prob_1[t]) * np.log(sigmoid(-np.dot(theta, contexts[t]))) 
                            for t in range(n_samples)])
        result = minimize(objective, self.theta.copy())
        theta_hat = result.x
        return skill, theta_hat

    def fit(self, ests: np.ndarray, contexts: np.ndarray):
        num_samples = ests.shape[0]
        assert ests.shape[1] == self.num_workers
        assert contexts.shape == (num_samples, self.context_len)
        # probability that the true label is 1
        prob_1 = np.zeros(num_samples)*np.nan
        for i in trange(self.max_iter):
            # E-step
            prob_1 = self.e_step(ests, contexts)
            # M-step
            self.skill, self.theta = self.m_step(ests, contexts, prob_1)
            # stopping criterion on skill change
            if i>0:
                skill_change = self.skill - old_skill
                if np.max(np.abs(skill_change)) < self.tol:
                    break
            old_skill = self.skill.copy()
        return
    
    def predict(self, ests: np.ndarray, contexts: np.ndarray):
        prob_1 = self.e_step(ests, contexts)
        labels = np.array(prob_1 > 0.5, dtype=int)
        return labels

    def predict_w_features_only(self, contexts: np.ndarray):
        prob_1 = sigmoid(np.dot(contexts, self.theta))
        labels = np.array(prob_1 > 0.5, dtype=int)
        return labels

class EMNeuralNetBinary:
    def __init__(self, seed: int, 
                 context_len: int, num_workers: int,
                 neural_net_cons: object,
                 max_iter: int = 100, tol: float = 1e-6,
                 lr: float = 0.01, batch_size: int = 32,
                 wt_decay: float = 1e-4, epochs: int = 10,
                 patience: int = 20):
        self.rng = np.random.default_rng(seed)
        self.context_len = context_len
        self.num_workers = num_workers
        # skill of each worker is more than 0.5 and is different across classes
        self.skill = (self.rng.random((num_workers,2))+1)/2
        self.max_iter = max_iter
        self.tol = tol
        # nn and training params
        self.neural_net_cons = neural_net_cons
        self.neural_net = neural_net_cons()
        self.lr = lr
        self.batch_size = batch_size
        self.wt_decay = wt_decay
        self.epochs = epochs
        self.patience = patience
    
    def e_step(self, ests: np.ndarray, contexts: np.ndarray):
        num_samples = ests.shape[0]
        contexts = torch.tensor(contexts).float()
        prob_1 = np.zeros(num_samples)*np.nan
        log_unnorm_probs = np.ones((num_samples, 2))*np.nan
        log_prior_prob_1 = np.log(sigmoid(self.neural_net(contexts).detach().numpy())).squeeze()
        ## logic for next line: 1-1/(1+exp(-x)) = 1/(1+exp(x)) = 1/(1+exp(-(-x)))
        log_prior_prob_0 = np.log(sigmoid(-self.neural_net(contexts).detach().numpy())).squeeze()
        log_unnorm_probs[:, 1] = ests@np.log(self.skill[:,1]) \
            + (1-ests)@np.log(1-self.skill[:,1]) + log_prior_prob_1
        log_unnorm_probs[:, 0] = (1-ests)@np.log(self.skill[:,0]) \
            + ests@np.log(1-self.skill[:,0]) + log_prior_prob_0
        log_prob_1 = log_unnorm_probs[:, 1] \
            - np.logaddexp(log_unnorm_probs[:, 0], log_unnorm_probs[:, 1])
        prob_1 = np.exp(log_prob_1)
        assert prob_1.shape == (num_samples,)
        try:
            assert np.all(prob_1 >= 0) and np.all(prob_1 <= 1)
        except:
            print("Error")
            idx = np.where((prob_1 < 0) | (prob_1 > 1))[0]
            print(idx)
            print(prob_1[idx])
            raise
        return prob_1
    
    def m_step(self, ests: np.ndarray, contexts: np.ndarray, 
               prob_1: np.ndarray):
        skill = np.zeros((self.num_workers,2))*np.nan
        for j in range(self.num_workers):
            arr = prob_1*ests[:, j] 
            skill[j,1] = np.mean(arr)/np.mean(prob_1)
            arr = (1-prob_1)*(1-ests[:, j])
            skill[j,0] = np.mean(arr)/np.mean(1-prob_1)
        try:
            assert np.all(skill >= 0) and np.all(skill <= 1)
        except:
            print("Error")
            print(skill)
            raise
        # self.train_neural_net(contexts, prob_1)
        num_samples = contexts.shape[0]
        features_train = torch.tensor(contexts[:int(0.8*num_samples)]).float()
        prob_1_train = torch.tensor(prob_1[:int(0.8*num_samples)]).float().unsqueeze(1)
        features_val = torch.tensor(contexts[int(0.8*num_samples):]).float()
        prob_1_val = torch.tensor(prob_1[int(0.8*num_samples):]).float().unsqueeze(1)
        self.neural_net, stat_dict = train_neural_net(neural_net=self.neural_net,
                                                      x_train=features_train,
                                                      y_train=prob_1_train,
                                                      x_val=features_val,
                                                      y_val=prob_1_val,
                                                      lr=self.lr, weight_decay=self.wt_decay,
                                                      epochs=self.epochs, patience=self.patience)
        self.skill = skill

    # def train_neural_net(self, contexts: np.ndarray, prob_1: np.ndarray):
    #     n_samples = contexts.shape[0]
    #     optimizer = optim.Adam(self.neural_net.parameters(), 
    #                            lr=self.lr, weight_decay=self.wt_decay)
    #     criterion = nn.BCEWithLogitsLoss()
    #     train_losses = []
    #     contexts = torch.tensor(contexts.copy()).float()
    #     for epoch in range(self.epochs):
    #         self.neural_net.train()
    #         for i in range(0, n_samples, self.batch_size):
    #             batch_features = contexts[i:i+self.batch_size]
    #             batch_prob_1 = prob_1[i:i+self.batch_size]
    #             outputs = self.neural_net(batch_features)
    #             loss = criterion(outputs, torch.tensor(batch_prob_1[:,None]).float())
    #             train_losses.append(loss.item())
    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    def fit(self, ests: np.ndarray, contexts: np.ndarray):
        num_samples = ests.shape[0]
        assert ests.shape[1] == self.num_workers
        assert contexts.shape == (num_samples, self.context_len)
        # probability that the true label is 1
        prob_1 = np.zeros(num_samples)*np.nan
        skill_changes = []
        for i in trange(self.max_iter):
            # E-step
            prob_1 = self.e_step(ests, contexts)
            # M-step
            self.m_step(ests, contexts, prob_1)
            # stopping criterion on skill change
            if i>0:
                skill_change = self.skill - old_skill
                skill_changes.append(np.max(np.abs(skill_change)))
                if np.max(np.abs(skill_change)) < self.tol:
                    break
            old_skill = self.skill.copy()
        return
    
    def predict(self, ests: np.ndarray, contexts: np.ndarray):
        prob_1 = self.e_step(ests, contexts)
        labels = np.array(prob_1 > 0.5, dtype=int)
        return labels
    
    def predict_w_features_only(self, contexts: np.ndarray):
        # prob_1 = sigmoid(self.neural_net(torch.tensor(contexts).float()).detach().numpy())
        # labels = np.array(prob_1 > 0.5, dtype=int)
        labels = np.array(self.neural_net(torch.tensor(contexts).float()).detach().numpy() > 0, dtype=int)
        labels = labels.flatten()
        return labels

class NeuralNetMajVote:
    def __init__(self,
                 num_workers: int,
                 context_len: int,
                 neural_net_cons: object,
                 lr: float=1e-3,
                 wt_decay: float=1e-4,
                 epochs: int=1000,
                 patience: int=20,
                 ) -> None:
        self.num_workers = num_workers
        self.context_len = context_len
        self.neural_net_cons = neural_net_cons
        self.lr = lr
        self.wt_decay = wt_decay
        self.epochs = epochs
        self.patience = patience

    def fit(self, contexts: np.ndarray, ests: np.ndarray):
        assert ests.shape[1] == self.num_workers
        assert contexts.shape[1] == self.context_len
        self.scaler = StandardScaler()
        contexts = self.scaler.fit_transform(contexts)
        print("number of training samples: ", contexts.shape[0])
        print(np.median(np.abs(np.mean(contexts, axis=0))))
        print(np.median(np.std(contexts, axis=0)))
        # apply majority vote to get labels
        targets = MajorityVote(ests.shape[1]).predict(ests).reshape(-1,1)
        print("frequency of 1s in targets: ", np.mean(targets))
        targets_train_ten = torch.tensor(targets).float()[:int(0.8*contexts.shape[0])]
        targets_val_ten = torch.tensor(targets).float()[int(0.8*contexts.shape[0]):]
        contexts_train_ten = torch.tensor(contexts).float()[:int(0.8*contexts.shape[0])]
        contexts_val_ten = torch.tensor(contexts).float()[int(0.8*contexts.shape[0]):]
        # targets_train_ten = torch.tensor(targets).float()
        # targets_val_ten = torch.tensor(targets).float()
        # contexts_train_ten = torch.tensor(contexts).float()
        # contexts_val_ten = torch.tensor(contexts).float()
        self.neural_net, self.stats_dict =\
              train_neural_net(neural_net=self.neural_net_cons(), x_train=contexts_train_ten,
                                y_train=targets_train_ten, x_val=contexts_val_ten,
                                y_val=targets_val_ten, lr=self.lr, weight_decay=self.wt_decay, 
                                epochs=self.epochs, patience=self.patience, testing=True)
    
    def predict(self, contexts: np.ndarray, ests: np.ndarray):
        contexts = self.scaler.transform(contexts)
        labels = np.array(self.neural_net(torch.tensor(contexts).float()).detach().numpy() > 0, dtype=int)
        labels = labels.flatten()
        return labels