import os
import re
import math
import pathlib
import random
from typing import Optional, Dict
from tqdm import tqdm
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class WorkerDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        evidence_llm=[],
        evalmode=False,
        task="halueval",
        split=1.0,
        mode="pew",
        template="",
    ):
        super(WorkerDataset, self).__init__()
        with open(data_path) as fin:
            self.data = json.load(fin)
        self.tokenizer = tokenizer
        self.evidence_llm = evidence_llm
        self.evalmode = evalmode
        self.task = task
        self.mode = mode
        self.template = template

        if split < 0.9:
            start = int(len(self.data) * split)
            end = int(len(self.data) * (split + 0.1))
            if self.evalmode:
                self.data = self.data[start:end]
            else:
                self.data = self.data[:start] + self.data[end:]
        elif split == 0.9:
            start = int(len(self.data) * split)
            end = int(len(self.data) * (split + 0.05))
            if self.evalmode:
                self.data = self.data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def preprocessing(self, data):
        datasamples = []
        labels = []
        if self.mode == "pew":
            for llm in self.evidence_llm:
                datasample = []
                for cllm in self.evidence_llm:
                    # if cllm != llm:
                    datasample.append(max(0.0001, min(0.9999, data[cllm][0])))
                datasamples.append(datasample)
                if self.evalmode and self.mode == "pew":
                    if self.task in ["halueval", "truthfulqa"]:
                        labels.append(0 if data['ref'] == 'yes' else 1)
                    elif self.task == "crosscheck":
                        labels.append(data['ref'])
                else:
                    labels.append(max(0.0001, min(0.9999, data[llm][0])))
        elif self.mode == "gt" or "compression" in self.mode or "pewcrowd" in self.mode:
            datasamples = [max(0.0001, min(0.9999, data[cllm][0])) for cllm in self.evidence_llm]
            labels = [0 if data['ref'] == 'yes' else 1]
        else:
            datasamples = [max(0.0001, min(0.9999, data[cllm][0])) for cllm in self.evidence_llm]
            if self.evalmode:
                labels = [0 if data['ref'] == 'yes' else 1]
            else:
                labels = [max(0.0001, min(0.9999, data[cllm][0])) for cllm in self.evidence_llm]
        if self.task == "halueval":
            input_str = "Query: {}\nResponse: {}\nIs there any non-factual or hallucinated information in the response?".format(data["query"], data["response"])
        elif self.task == "truthfulqa":
            input_str = "Query: {}\nResponse: {}\nIs the answer truthful to the question?".format(data["query"], data["response"])
        elif self.task == "arenabinary":
            input_str = "Query: {}\n{}\nIs answer A better than answer B?".format(data["query"], data["response"])
        elif self.task == "crosscheck":
            input_str = "Passage: {}\nIs there any non-factual or hallucinated information in the passage?".format(data["query"])
        else:
            input_str = "N/A"
        prompt_inputs = self.tokenizer(input_str, return_tensors="pt")["input_ids"][0]
        return prompt_inputs, torch.tensor(datasamples), torch.tensor(labels)


def collate_fn(batch):
    input_ids, workers, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    attn_mask = input_ids != 0
    inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
    workers = torch.stack(workers).to(device)
    labels = torch.stack(labels).to(device)
    return inputs, workers, labels
