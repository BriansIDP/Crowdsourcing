import numpy as np
from tqdm import trange

class EMSymmetricBinary:
    
    def __init__(self, seed: int, 
                 num_models: int,
                 skill_init: np.ndarray = None,
                 tol: float = 1e-8, max_iter: int = 100):
        self.rng = np.random.default_rng(seed)
        self.num_models = num_models
        # self.pi = np.ones(num_classes) / num_classes
        # skill: probability of answering correctly
        if skill_init is not None:
            self.skill = skill_init
        else:
            self.skill = (self.rng.random(num_models)+1)/2
        assert np.all(self.skill >= 0.5) and np.all(self.skill <= 1)
        # assert self.theta.shape == (2, num_models)
        self.epsilon = tol
        self.max_iter = max_iter
    
    def fit(self, ests: np.ndarray):
        num_samples = ests.shape[0]
        assert ests.shape[1] == self.num_models
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
        skill = np.zeros(self.num_models)*np.nan
        for j in range(self.num_models):
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
