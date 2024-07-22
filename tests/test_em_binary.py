import pytest
import numpy as np

from worker_aggregation import EMSymmetricBinary, EMAsymmetricBinary

@pytest.fixture
def synthetic_data_sym():
    def _synthetic_data(seed: int, num_samples: int, num_workers: int):
        rng = np.random.default_rng(seed)
        skill = 0.5 + 0.5*rng.random(num_workers)
        ests = np.zeros((num_samples, num_workers), dtype=np.int32)
        true_labels = np.zeros(num_samples, dtype=np.int32)
        for i in range(num_samples):
            true_label = rng.integers(0, 2)
            true_labels[i] = true_label
            for j in range(num_workers):
                if true_label == 1:
                    ests[i, j] = rng.choice([0, 1], p=[1-skill[j], skill[j]])
                else:
                    ests[i, j] = rng.choice([0, 1], p=[skill[j], 1-skill[j]])
        return ests, true_labels, skill
    return _synthetic_data

@pytest.fixture
def synthetic_data_asym():
    def _synthetic_data(seed: int, num_samples: int, num_workers: int):
        rng = np.random.default_rng(seed)
        skill = 0.5 + 0.5*rng.random((num_workers, 2))
        ests = np.zeros((num_samples, num_workers), dtype=np.int32)
        true_labels = np.zeros(num_samples, dtype=np.int32)
        for i in range(num_samples):
            true_label = rng.integers(0, 2)
            true_labels[i] = true_label
            for j in range(num_workers):
                if true_label == 1:
                    ests[i, j] = rng.choice([0, 1], p=[1-skill[j,1], skill[j,1]])
                else:
                    ests[i, j] = rng.choice([0, 1], p=[skill[j,0], 1-skill[j,0]])
        return ests, true_labels, skill
    return _synthetic_data

class TestEMSymmetricBinary:
    def test_single_sample(self,):
        skill_init = np.array([0.6, 0.7, 0.8])
        em_model = EMSymmetricBinary(seed=42, num_workers=3, skill_init=skill_init)
        ests = np.array([[0, 1, 1]])
        exp_prob_1 = 0.4*0.7*0.8/(0.4*0.7*0.8 + 0.6*0.3*0.2)
        prob_1 = em_model.e_step(ests)
        assert np.isclose(prob_1, exp_prob_1)
        print("E-step test passed")

        ## M-step test
        exp_skill = np.array([1-prob_1[0], prob_1[0], prob_1[0]])
        skill = em_model.m_step(ests, prob_1)
        # print(exp_skill)
        # print(skill)
        assert np.allclose(skill, exp_skill)
        print("M-step test passed")
    
    def test_synthetic_data(self, synthetic_data_sym):
        num_samples = 10000
        num_workers = 5
        ests, _, skill = synthetic_data_sym(seed=42, num_samples=num_samples, num_workers=num_workers)
        em_model = EMSymmetricBinary(seed=42, num_workers=num_workers)
        em_model.fit(ests)
        logit_skill = np.log(skill) - np.log(1-skill)
        logit_em_skill = np.log(em_model.skill) - np.log(1-em_model.skill)
        assert np.allclose(logit_skill, logit_em_skill, atol=1e-1)
        print("true logit skill", logit_skill)
        print("EM logit skill", logit_em_skill)
        print("Synthetic data test passed")

class TestEMAsymmetricBinary:
    def test_sym_synthetic_data(self, synthetic_data_sym):
        num_samples = 10000
        num_workers = 5
        ests, _, skill = synthetic_data_sym(seed=42, num_samples=num_samples, num_workers=num_workers)
        em_model = EMAsymmetricBinary(seed=42, num_workers=num_workers)
        em_model.fit(ests)
        skill_full = np.ones((num_workers, 2))*skill[:, None]
        logit_skill = np.log(skill_full) - np.log(1-skill_full)
        logit_em_skill = np.log(em_model.skill) - np.log(1-em_model.skill)
        assert np.allclose(logit_skill, logit_em_skill, atol=1e-1)
        print("true logit skill", logit_skill)
        print("EM logit skill", logit_em_skill)
        print("Symmetric synthetic data test passed")
    
    def test_asym_synthetic_data(self, synthetic_data_asym):
        num_samples = 10000
        num_workers = 3
        ests, _, skill = synthetic_data_asym(seed=42, num_samples=num_samples, num_workers=num_workers)
        em_model = EMAsymmetricBinary(seed=42, num_workers=num_workers)
        em_model.fit(ests)
        logit_skill = np.log(skill) - np.log(1-skill)
        logit_em_skill = np.log(em_model.skill) - np.log(1-em_model.skill)
        assert np.allclose(logit_skill, logit_em_skill, atol=5e-1)
        print("true logit skill", logit_skill)
        print("EM logit skill", logit_em_skill)
        print("Asymmetric synthetic data test passed")

    def test_reg_m_step(self, synthetic_data_asym):
        num_samples = 10
        num_workers = 3
        reg_m_step = 1000
        ests, _, _ = synthetic_data_asym(seed=42, num_samples=num_samples, 
                                             num_workers=num_workers)
        em_model = EMAsymmetricBinary(seed=42, num_workers=num_workers, reg_m_step=reg_m_step)
        em_model.fit(ests)
        exp_skill = 0.5*np.ones((num_workers, 2))
        logit_skill = np.log(exp_skill) - np.log(1-exp_skill)
        logit_em_skill = np.log(em_model.skill) - np.log(1-em_model.skill)
        assert np.allclose(logit_skill, logit_em_skill, atol=5e-1)
        print("true logit skill", logit_skill)
        print("EM logit skill", logit_em_skill)
        print("Asymmetric synthetic data test passed")