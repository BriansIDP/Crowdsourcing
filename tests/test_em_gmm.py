import pytest
import numpy as np

from worker_agg import EM_GMM

@pytest.fixture
def synthetic_data1():
    def _synthetic_data(seed: int, num_samples: int, num_workers: int):
        rng = np.random.default_rng(seed)
        cov_mat0 = np.diag([1, 2, 1.5])
        cov_mat1 = np.diag([1.5, 1, 2])
        cov_mats = np.concatenate([cov_mat0[None, :, :], cov_mat1[None, :, :]], axis=0)
        num_classes = 2
        means = 2*np.array([[-1, -1, -1], [1, 1, 1]])
        ests = np.zeros((num_samples, num_workers))
        true_labels = np.zeros(num_samples, dtype=np.int32)
        for i in range(num_samples):
            true_label = rng.integers(0, num_classes)
            true_labels[i] = true_label
            ests[i] = rng.multivariate_normal(mean=means[true_label], 
                                            cov=cov_mats[true_label])
        return ests, true_labels, means, cov_mats
    return _synthetic_data

@pytest.fixture
def synthetic_data2():
    def _synthetic_data(seed: int, num_samples: int, num_workers: int):
        rng = np.random.default_rng(seed)
        cov_mat0 = np.diag([1, 2, 1.5])
        cov_mat1 = np.diag([1.5, 1, 2])
        cov_mats = np.concatenate([cov_mat0[None, :, :], cov_mat1[None, :, :]], axis=0)
        num_classes = 2
        mean0 = rng.normal(size=num_workers) - 2
        mean1 = rng.normal(size=num_workers) + 2
        means = np.array([mean0, mean1])
        ests = np.zeros((num_samples, num_workers))
        true_labels = np.zeros(num_samples, dtype=np.int32)
        for i in range(num_samples):
            true_label = rng.integers(0, num_classes)
            true_labels[i] = true_label
            ests[i] = rng.multivariate_normal(mean=means[true_label], 
                                            cov=cov_mats[true_label])
        return ests, true_labels, means, cov_mats
    return _synthetic_data

class TestEMGMM:
    def test_synthetic_data1(self, synthetic_data1):
        num_samples = 1000
        num_workers = 3
        ests, true_labels, means, cov_mats = synthetic_data1(seed=42, num_samples=num_samples, num_workers=num_workers)
        var_of_means = 5
        em_model = EM_GMM(num_workers=num_workers, cov_mat_diag=1, )
        em_model.fit(ests)
        preds = em_model.predict(ests)
        accuracy = np.mean(preds == true_labels)
        print("Accuracy:", accuracy)
        assert accuracy > 0.9
        print("true mean 0", means[0])
        print("em mean 0", em_model.mean0)
        print("true cov 0", cov_mats[0])
        print("em cov 0", em_model.cov0)

    def test_synthetic_data2(self, synthetic_data2):
        num_samples = 1000
        num_workers = 3
        ests, true_labels, means, cov_mats = synthetic_data2(seed=42, num_samples=num_samples, num_workers=num_workers)
        var_of_means = 5
        em_model = EM_GMM(num_workers=num_workers, cov_mat_diag=1, )
        em_model.fit(ests)
        preds = em_model.predict(ests)
        accuracy = np.mean(preds == true_labels)
        print("Accuracy:", accuracy)
        assert accuracy > 0.9
        print("true mean 0", means[0])
        print("em mean 0", em_model.mean0)
        print("true cov 0", cov_mats[0])
        print("em cov 0", em_model.cov0)
