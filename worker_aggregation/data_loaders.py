import json
from pathlib import Path

import numpy as np
import torch
from .utils import TwoLayerMLP

class HaluDialogueBinary:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        est_dict = {}
        for model in self.model_list:
            hits = 0
            est_dict[model] = []
            outcomes = []
            filepath = Path(self.datapath) / "halueval_dialogue_{}.json".format(model)
            with open(filepath) as fin:
                modeldata = json.load(fin)[model]
            for datapiece in modeldata:
                outcome = 0 if datapiece["ref"] == "yes" else 1
                outcomes.append(outcome)
                # est_dict[model].append(datapiece["prob"])
                est = np.argmax(datapiece["prob"])
                est_dict[model].append(est)
                # if datapiece["prob"][0] > datapiece["prob"][1] and outcome == 0:
                #     hits += 1
                # elif datapiece["prob"][1] > datapiece["prob"][0] and outcome == 1:
                #     hits += 1
            # print("{} Acc: {:.3f}".format(model, hits/len(est_dict[model])))
        ests = np.zeros((len(outcomes), len(self.model_list)))
        for i, model in enumerate(self.model_list):
            ests[:, i] = est_dict[model]
        return ests, outcomes

class HaluDialogueLogit:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        est_dict = {}
        for model in self.model_list:
            hits = 0
            est_dict[model] = []
            outcomes = []
            filepath = Path(self.datapath) / "halueval_dialogue_{}.json".format(model)
            with open(filepath) as fin:
                modeldata = json.load(fin)[model]
            for datapiece in modeldata:
                outcome = 0 if datapiece["ref"] == "yes" else 1
                outcomes.append(outcome)
                # est_dict[model].append(datapiece["prob"])
                # est = np.argmax(datapiece["prob"])
                est = np.log(datapiece["prob"][1] / datapiece["prob"][0])
                est_dict[model].append(est)
                # if datapiece["prob"][0] > datapiece["prob"][1] and outcome == 0:
                #     hits += 1
                # elif datapiece["prob"][1] > datapiece["prob"][0] and outcome == 1:
                #     hits += 1
            # print("{} Acc: {:.3f}".format(model, hits/len(est_dict[model])))
        ests = np.zeros((len(outcomes), len(self.model_list)))
        for i, model in enumerate(self.model_list):
            ests[:, i] = est_dict[model]
        return ests, outcomes

class HaluQABinary:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        est_dict = {}
        for model in self.model_list:
            hits = 0
            est_dict[model] = []
            outcomes = []
            filepath = Path(self.datapath) / "halueval_qa_{}.json".format(model)
            with open(filepath) as fin:
                modeldata = json.load(fin)[model]
            for datapiece in modeldata:
                outcome = 0 if datapiece["ref"] == "yes" else 1
                outcomes.append(outcome)
                # est_dict[model].append(datapiece["prob"])
                est = np.argmax(datapiece["prob"])
                est_dict[model].append(est)
                if datapiece["prob"][0] > datapiece["prob"][1] and outcome == 0:
                    hits += 1
                elif datapiece["prob"][1] > datapiece["prob"][0] and outcome == 1:
                    hits += 1
            # print("{} Acc: {:.3f}".format(model, hits/len(est_dict[model])))
        ests = np.zeros((len(outcomes), len(self.model_list)))
        for i, model in enumerate(self.model_list):
            ests[:, i] = est_dict[model]
        return ests, outcomes

class HaluDialBertPCA:
    def __init__(self, filepath, model_list):
        assert Path(filepath).exists()
        self.all_data = np.load(filepath)
        self.num_workers = len(model_list)

    def get_data(self, split_type='train'):
        if split_type == 'train':
            context = self.all_data['context_train']
            ests = self.all_data['ests_train']
            outcomes = self.all_data['outcomes_train']
        elif split_type == 'val':
            context = self.all_data['context_val']
            ests = self.all_data['ests_val']
            outcomes = self.all_data['outcomes_val']
        elif split_type == 'test':
            context = self.all_data['context_test']
            ests = self.all_data['ests_test']
            outcomes = self.all_data['outcomes_test']
        else:
            raise ValueError("Invalid split type")
        assert context.shape[0] == ests.shape[0]
        assert ests.shape[1] == self.num_workers
        return context, ests, outcomes

class HaluDialBertEmbed:
    def __init__(self, filepath, model_list):
        assert Path(filepath).exists()
        self.all_data = np.load(filepath)
        self.num_workers = len(model_list)

    def get_data(self, split_type='train'):
        if split_type == 'train':
            context = self.all_data['context_train']
            ests = self.all_data['ests_train']
            outcomes = self.all_data['outcomes_train']
        elif split_type == 'val':
            context = self.all_data['context_val']
            ests = self.all_data['ests_val']
            outcomes = self.all_data['outcomes_val']
        elif split_type == 'test':
            context = self.all_data['context_test']
            ests = self.all_data['ests_test']
            outcomes = self.all_data['outcomes_test']
        else:
            raise ValueError("Invalid split type")
        assert context.shape[0] == ests.shape[0]
        assert ests.shape[1] == self.num_workers
        return context, ests, outcomes

class SynTwoLayerMLPData:
    def __init__(self, seed: int, num_features: int, 
                 num_workers: int, hidden_size: int=10, temp: float=1.0):
        self.num_features = num_features
        self.num_workers = num_workers
        self.rng = np.random.default_rng(seed)
        self.skill = self.rng.uniform(0.55, 0.8, size=(num_workers,2))
        self.hidden_size = hidden_size
        self.temp = temp
        self.seed = seed
    
    def gen_data(self, num_samples: int, testing=False):
        features = self.rng.normal(size=(num_samples, self.num_features))
        true_nn = TwoLayerMLP(self.seed, self.num_features, self.hidden_size)
        features_tensor = torch.tensor(features).float()
        outputs = true_nn(features_tensor).detach().numpy().flatten()
        outputs = (outputs - np.mean(outputs))/self.temp
        logit = lambda x: 1/(1+np.exp(-x))
        outcomes = np.array(logit(self.rng.random(num_samples)) < outputs, 
                                dtype=int)
        assert outcomes.shape == (num_samples,)
        flip = self.rng.random((num_samples, self.num_workers)) > self.skill.T[outcomes,:]
        flip = flip.astype(float)
        ests = outcomes[:,None]*1.0 + flip*(1.0-2*outcomes[:,None])
        if testing:
            return ests, features, outcomes, self.skill
        return ests, features, outcomes

class SynLogisticData:
    def __init__(self, seed: int, num_features: int, 
                 num_workers: int, temp: float=1.0):
        self.num_features = num_features
        self.num_workers = num_workers
        self.rng = np.random.default_rng(seed)
        self.skill = self.rng.uniform(0.55, 0.8, size=(num_workers,2))
        self.theta = self.rng.normal(size=num_features)
        self.temp = temp
        self.seed = seed
    
    def gen_data(self, num_samples: int, testing=False):
        features = self.rng.normal(size=(num_samples, self.num_features))
        sigmoid = lambda x: 1/(1+np.exp(-x))
        outcomes = np.array(self.rng.random(num_samples) < sigmoid(features@self.theta/self.temp), 
                            dtype=int)
        flip = self.rng.random((num_samples, self.num_workers)) > self.skill.T[outcomes,:]
        flip = flip.astype(float)
        ests = outcomes[:,None]*1.0 + flip*(1.0-2*outcomes[:,None])
        if testing:
            return ests, features, outcomes, self.skill, self.theta
        return ests, features, outcomes