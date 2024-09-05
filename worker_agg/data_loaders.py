import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from .utils import TwoLayerMLP

class HaluDialogueBinary:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        est_dict = {}
        for model in self.model_list:
            est_dict[model] = []
            outcomes = []
            filepath = Path(self.datapath) / "halueval_dialogue_{}.json".format(model)
            with open(filepath) as fin:
                modeldata = json.load(fin)[model]
            for datapiece in modeldata:
                outcome = 0 if datapiece["ref"] == "yes" else 1
                outcomes.append(outcome)
                est = np.argmax(datapiece["prob"])
                est_dict[model].append(est)
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

class HaluDialogueProb:
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
                est = datapiece["prob"][1]
                est_dict[model].append(est)
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

class HaluDialEmbed:
    def __init__(self, filepath, model_list, 
                 seed: int):
        assert Path(filepath).exists()
        self.all_data = np.load(filepath)
        self.num_workers = len(model_list)
        self.rng = np.random.default_rng(seed)

    def get_data(self, split=0.5, split_type='train',
                 shuffle=False):
        contexts = self.all_data['embeddings']
        ests = self.all_data['ests']
        outcomes = self.all_data['outcomes']
        total_samples = contexts.shape[0]
        if split_type == 'train':
            contexts = contexts[:int(split*total_samples)]
            ests = ests[:int(split*total_samples)]
            outcomes = outcomes[:int(split*total_samples)]
        elif split_type == 'val':
            contexts = contexts[int(split*total_samples):]
            ests = ests[int(split*total_samples):]
            outcomes = outcomes[int(split*total_samples):]
        else:
            raise ValueError("Invalid split type")
        assert contexts.shape[0] == ests.shape[0]
        assert ests.shape[1] == self.num_workers
        if shuffle:
            shuffleidx = self.rng.permutation(contexts.shape[0])
            contexts = contexts[shuffleidx]
            ests = ests[shuffleidx]
            outcomes = outcomes[shuffleidx]
        return contexts, ests, outcomes

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


class HaluDialBinaryLM(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        model_path,
        evidence_llm=[],
        evalmode=False,
        split=0.5,
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
        with_gt=False,
        task='halueval',
    ):
        super().__init__()
        with open(data_path) as fin:
            self.data = json.load(fin)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.evidence_llm = evidence_llm
        self.evalmode = evalmode
        self.device = device

        # if split < 1.0:
        #     portion = int(len(self.data) * split)
        #     if self.evalmode:
        #         self.data = self.data[portion:] if portion > 0 else self.data[:portion]
        #         # self.data = self.data[portion:2*portion] 
        #     else:
        #         self.data = self.data[:portion] if portion > 0 else self.data[portion:]
        #         # self.data = self.data[:portion] 
        if split < 0.9:
            start = int(len(self.data) * split)
            end = int(len(self.data) * (split + 0.1))
            if self.evalmode:
                self.data = self.data[start:end]
            else:
                self.data = self.data[:start] + self.data[end:]

        self.with_gt = with_gt
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def preprocessing(self, data):
        ests = [data[cllm][0]<0.5 for cllm in self.evidence_llm]
        outcomes = [0 if data['ref'] == 'yes' else 1]
        if self.task == 'halueval':
            input_str = "Query: {}\nResponse: {}\nIs there any non-factual or hallucinated information in the response?".format(data["query"], data["response"])
        elif self.task == 'truthfulqa':
            input_str = "Query: {}\nResponse: {}\nIs the answer truthful to the question?".format(data["query"], data["response"])
        else:   
            raise ValueError(f"Invalid task {self.task}")
        prompt_inputs = self.tokenizer(input_str, return_tensors="pt")["input_ids"][0]
        if self.with_gt:
            return prompt_inputs, torch.tensor(ests).float(), torch.tensor(outcomes)
        else:
            return prompt_inputs, torch.tensor(ests).float()

    def collate_fn(self, batch):
        if self.with_gt:
            input_ids, ests, outcomes = zip(*batch)
            outcomes = torch.stack(outcomes).to(self.device)
        else:
            input_ids, ests = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        ests = torch.stack(ests).to(self.device)
        if self.with_gt:
            return inputs, ests, outcomes
        else:
            return inputs, ests

class TruthfulQABinary:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        filepath = Path(self.datapath) / "truthful_qa.json"
        with open(filepath) as fin:
            data = json.load(fin)
        ests = []
        outcomes = []
        for datap in data:
            ests.append([datap[cllm][0]>0.5 for cllm in self.model_list])
            outcomes.append(1 if datap['ref'] == 'yes' else 0)
        ests = np.array(ests)
        outcomes = np.array(outcomes)
        return ests, outcomes