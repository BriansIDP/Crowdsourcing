import numpy as np
from tqdm import trange

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
                 skill_init: np.ndarray = None,
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