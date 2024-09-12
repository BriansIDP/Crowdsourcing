import pytest
import numpy as np
import torch

from worker_agg import TwoLayerMLP, EMNeuralNetBinary, MajorityVote
from worker_agg import EMAsymmetricBinary

@pytest.fixture
def synthetic_data_nn():
    def _generate_data(seed: int=0, num_samples: int=1000, num_workers: int=5, 
                    context_len: int=10, hidden_size: int=10,
                    temp: float=1):
        rng = np.random.default_rng(seed)
        skill = rng.uniform(0.55, 0.8, size=(num_workers,2))
        contexts = rng.normal(size=(num_samples, context_len))
        true_nn = TwoLayerMLP(seed, context_len, hidden_size)
        features_tensor = torch.tensor(contexts).float()
        outputs = true_nn(features_tensor).detach().numpy().flatten()
        outputs = (outputs - np.mean(outputs))/temp
        sigmoid = lambda x: 1/(1+np.exp(-x))
        true_labels = np.array(rng.random(num_samples) < sigmoid(outputs), 
                                dtype=int)
        assert true_labels.shape == (num_samples,)
        flip = rng.random((num_samples, num_workers)) > skill.T[true_labels,:]
        flip = flip.astype(float)
        ests = true_labels[:,None]*1.0 + flip*(1.0-2*true_labels[:,None])
        return {'ests':ests, 'true_labels':true_labels, 'skill':skill, 'contexts':contexts}
    return _generate_data

class TestEMNNBinary:
    def test_em_feature_no_feature(self, synthetic_data_nn):
        # generate data
        train_size = 10000
        test_size = 1000
        num_samples = train_size + test_size
        hidden_size = 10
        temp = 0.5
        data_dict = synthetic_data_nn(num_samples=num_samples,
                                hidden_size=hidden_size, temp=temp)
        ests_train = data_dict['ests'][:train_size]
        ests_test = data_dict['ests'][train_size:]
        features_train = data_dict['contexts'][:train_size]
        features_test = data_dict['contexts'][train_size:]
        true_labels_train = data_dict['true_labels'][:train_size]
        true_labels_test = data_dict['true_labels'][train_size:]
        skill = data_dict['skill']
        num_workers = ests_train.shape[1]
        context_len = features_train.shape[1]

        # class imbalance
        print("class imbalance: ", np.mean(true_labels_train), np.mean(true_labels_test))

        ## EM with neural net
        policy_seed = 42
        def neural_net_cons():
            return TwoLayerMLP(policy_seed, context_len, 2*hidden_size)
        lr = 1e-3
        batch_size = train_size
        wt_decay = 1e-4
        epochs = 100
        em_model = EMNeuralNetBinary(seed=policy_seed, context_len=context_len, 
                                    num_workers=num_workers, neural_net_cons=neural_net_cons, 
                                    lr=lr, batch_size=batch_size,
                                    wt_decay=wt_decay, epochs=epochs)
        # fit the model
        em_model.fit(ests_train, features_train)
        print("true skill: ", skill)
        print("estimated skill: ", em_model.skill)
        assert np.allclose(em_model.skill, skill, atol=1e-1)
        print("skill of EMNeuralNetBinary matches true skill")

        ## EM without neural net
        em_model2 = EMAsymmetricBinary(seed=policy_seed, num_workers=num_workers)
        em_model2.fit(ests_train)
        assert np.allclose(em_model.skill, skill, atol=1e-1)
        print("skill of EMAssymetricBinary matches true skill")

        ## accuracy on train
        preds = em_model.predict(ests_train, features_train)
        acc = np.mean(preds == true_labels_train)
        preds = em_model2.predict(ests_train)
        acc2 = np.mean(preds == true_labels_train)
        preds = MajorityVote(num_workers).predict(ests_train)
        acc3 = np.mean(preds == true_labels_train)
        print("train accuracies are: ", acc, acc2, acc3)
        assert acc >= acc2, "EMNeuralNetBinary should be at least as good as EMAsymmetricBinary"
        assert acc >= acc3, "EMNeuralNetBinary should be at least as good as MajorityVote"
        print("train accuracy with contexts is better than without contexts")

        ## accuracy on test
        preds = em_model.predict(ests_test, features_test)
        acc = np.mean(preds == true_labels_test)
        preds = em_model2.predict(ests_test)
        acc2 = np.mean(preds == true_labels_test)
        preds = MajorityVote(num_workers).predict(ests_test)
        acc3 = np.mean(preds == true_labels_test)
        print("test accuracies are: ", acc, acc2, acc3)
        assert acc >= acc2, "EMNeuralNetBinary should be at least as good as EMAsymmetricBinary"
        assert acc >= acc3, "EMNeuralNetBinary should be at least as good as MajorityVote"
        print("test accuracy with contexts is better than without contexts")

    # def test_feature_only_perf(self, synthetic_data_nn):
    #     # generate data
    #     train_size = 10000
    #     test_size = 1000
    #     num_samples = train_size + test_size
    #     hidden_size = 10
    #     temp = 0.5
    #     data_dict = synthetic_data_nn(num_samples=num_samples,
    #                             hidden_size=hidden_size, temp=temp)
    #     ests_train = data_dict['ests'][:train_size]
    #     ests_test = data_dict['ests'][train_size:]
    #     features_train = data_dict['contexts'][:train_size]
    #     features_test = data_dict['contexts'][train_size:]
    #     true_labels_train = data_dict['true_labels'][:train_size]
    #     true_labels_test = data_dict['true_labels'][train_size:]
    #     skill = data_dict['skill']
    #     num_workers = ests_train.shape[1]
    #     context_len = features_train.shape[1]

    #     # class imbalance
    #     print("class imbalance: ", np.mean(true_labels_train), np.mean(true_labels_test))

    #     ## EM with neural net
    #     policy_seed = 42
    #     neural_net = TwoLayerMLP(policy_seed, context_len, 2*hidden_size)
    #     lr = 1e-3
    #     batch_size = train_size
    #     wt_decay = 1e-4
    #     epochs = 10
    #     em_model = EMNeuralNetBinary(seed=policy_seed, context_len=context_len, 
    #                                 num_workers=num_workers, neural_net=neural_net, 
    #                                 lr=lr, batch_size=batch_size,
    #                                 wt_decay=wt_decay, epochs=epochs)
    #     # fit the model
    #     em_model.fit(ests_train, features_train)
    #     assert np.allclose(em_model.skill, skill, atol=1e-1)
    #     print("skill of EMNeuralNetBinary matches true skill")
    #     # train a 2-layer MLP with majority vote predictions
    #     targets = MajorityVote(num_workers).predict(ests_train).reshape(-1,1)
    #     maj_model = TwoLayerMLP(policy_seed, context_len, 2*hidden_size)
    #     lr = 1e-3
    #     wt_decay = 1e-4
    #     epochs = 10
    #     optimizer = optim.Adam(maj_model.parameters(), 
    #                            lr=lr, weight_decay=wt_decay)
    #     criterion = nn.BCEWithLogitsLoss()
    #     features_train = torch.tensor(features_train.copy()).float()
    #     for epoch in range(epochs):
    #         # forward pass
    #         maj_model.train()
    #         outputs = maj_model(features_train)
    #         loss = criterion(outputs, torch.tensor(targets).float())

    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     # accuracy on train
    #     preds = em_model.predict_w_features_only(features_train)
    #     acc = np.mean(preds == true_labels_train)
    #     preds = maj_model(features_train).detach().numpy().flatten() > 0
    #     acc2 = np.mean(preds == true_labels_train)
    #     print("train accuracies are: ", acc, acc2)
    #     assert acc >= acc2, "EMLogisticBinary should be at least as good as 2-layer MLP with MajorityVote"
    #     print("train accuracy with EMLogisticBinary is better than with 2-layer MLP with MajorityVote")

    #     # accuracy on test
    #     preds = em_model.predict_w_features_only(features_test)
    #     acc = np.mean(preds == true_labels_test)
    #     features_test = torch.tensor(features_test.copy()).float()
    #     preds = maj_model(features_test).detach().numpy().flatten() > 0
    #     acc2 = np.mean(preds == true_labels_test)
    #     print("test accuracies are: ", acc, acc2)
    #     assert acc >= acc2, "EMLogisticBinary should be at least as good as 2-layer MLP with MajorityVote"
    #     print("test accuracy with EMLogisticBinary is better than with 2-layer MLP with MajorityVote")

