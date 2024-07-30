import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from worker_aggregation import EMLogisticBinary, EMAsymmetricBinary
from worker_aggregation import MajorityVote

sigmoid = lambda x: 1/(1+np.exp(-x))

@pytest.fixture
def synthetic_data_logistic():
    def _synthetic_data(seed: int=0, num_features: int=10, num_workers: int=10,
                        num_samples: int=1000, temp: float=1.0):
        rng = np.random.default_rng(seed)
        # skill = (rng.random((num_workers,2))+1)/2
        skill = rng.uniform(0.55, 0.8, size=(num_workers,2))
        features = rng.normal(size=(num_samples, num_features))
        theta = rng.normal(size=num_features)
        true_labels = np.array(rng.random(num_samples) < sigmoid(features@theta/temp), 
                            dtype=int)
        flip = rng.random((num_samples, num_workers)) > skill.T[true_labels,:]
        flip = flip.astype(float)
        ests = true_labels[:,None]*1.0 + flip*(1.0-2*true_labels[:,None])
        return {'ests':ests, 'true_labels':true_labels, 'skill':skill, 'theta':theta,
                'features':features}
    return _synthetic_data

class TestEMLogisticBinary:
    def test_em_feature_no_feature(self, synthetic_data_logistic):
        # sampling data
        train_size = 1000
        test_size = 100
        num_samples = train_size + test_size
        num_workers = 5
        temp = 0.5
        data_dict = synthetic_data_logistic(num_workers=num_workers,
                                        num_samples=num_samples, temp=temp)
        ests = data_dict['ests']
        ests_test = ests[train_size:]
        ests = ests[:train_size]
        features = data_dict['features']
        features_test = features[train_size:]
        features = features[:train_size]
        true_labels = data_dict['true_labels']
        true_labels_test = true_labels[train_size:]
        true_labels = true_labels[:train_size]
        theta = data_dict['theta']
        skill = data_dict['skill']
        n_workers = ests.shape[1]
        n_features = theta.shape[0]

        # train em with features
        em_model = EMLogisticBinary(seed=0, n_features=n_features, num_workers=n_workers)
        em_model.fit(ests, features)
        assert np.allclose(em_model.skill, skill, atol=1e-1)
        # assert nl.norm(em_model.theta - theta)/n_features**0.5 < 1e-1
        # train em without features
        em_model2 = EMAsymmetricBinary(seed=0, num_workers=n_workers)
        em_model2.fit(ests)

        # accuracy on train
        preds = em_model.predict(ests, features)
        acc = np.mean(preds == true_labels)
        preds = em_model2.predict(ests)
        acc2 = np.mean(preds == true_labels)
        preds = MajorityVote(num_workers).predict(ests)
        acc3 = np.mean(preds == true_labels)
        print("train accuracies are: ", acc, acc2, acc3)
        assert acc >= acc2, "EMLogisticBinary should be at least as good as EMAsymmetricBinary"
        assert acc >= acc3, "EMLogisticBinary should be at least as good as MajorityVote"
        print("train accuracy with features is better than without features")

        # accuracy on test
        preds = em_model.predict(ests_test, features_test)
        acc = np.mean(preds == true_labels_test)
        preds = em_model2.predict(ests_test)
        acc2 = np.mean(preds == true_labels_test)
        preds = MajorityVote(num_workers).predict(ests_test)
        acc3 = np.mean(preds == true_labels_test)
        print("test accuracies are: ", acc, acc2, acc3)
        assert acc >= acc2, "EMLogisticBinary should be at least as good as EMAsymmetricBinary"
        assert acc >= acc3, "EMLogisticBinary should be at least as good as MajorityVote"
        print("test accuracy with features is better than without features")
    
    def test_feature_only_perf(self, synthetic_data_logistic):
        # sampling data
        train_size = 1000
        test_size = 100
        num_samples = train_size + test_size
        num_workers = 5
        temp = 0.5
        data_dict = synthetic_data_logistic(num_workers=num_workers,
                                        num_samples=num_samples, temp=temp)
        ests = data_dict['ests']
        ests_test = ests[train_size:]
        ests = ests[:train_size]
        features = data_dict['features']
        features_test = features[train_size:]
        features = features[:train_size]
        true_labels = data_dict['true_labels']
        true_labels_test = true_labels[train_size:]
        true_labels = true_labels[:train_size]
        theta = data_dict['theta']
        skill = data_dict['skill']
        n_workers = ests.shape[1]
        n_features = theta.shape[0]

        # train em with features
        em_model = EMLogisticBinary(seed=0, n_features=n_features, num_workers=n_workers)
        em_model.fit(ests, features)
        assert np.allclose(em_model.skill, skill, atol=1e-1)
        # train logistic regression with majority vote predictions
        targets = MajorityVote(num_workers).predict(ests)
        maj_model = LogisticRegression()
        maj_model.fit(features, targets)

        # accuracy on train
        preds = em_model.predict_w_features_only(features)
        acc = np.mean(preds == true_labels)
        preds = maj_model.predict(features)
        acc2 = np.mean(preds == true_labels)
        print("train accuracies are: ", acc, acc2)
        assert acc >= acc2, "EMLogisticBinary should be at least as good as logistic reg with MajorityVote"
        print("train accuracy with EMLogisticBinary is better than with logistic reg with MajorityVote")

        # accuracy on test
        preds = em_model.predict_w_features_only(features_test)
        acc = np.mean(preds == true_labels_test)
        preds = maj_model.predict(features_test)
        acc2 = np.mean(preds == true_labels_test)
        print("test accuracies are: ", acc, acc2)
        assert acc >= acc2, "EMLogisticBinary should be at least as good as logistic reg with MajorityVote"
        print("test accuracy with EMLogisticBinary is better than with logistic reg with MajorityVote")
