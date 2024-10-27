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
            end = int(len(self.data) * split)
            start = int(len(self.data) * 0.9)
            # end = int(len(self.data) * (split + 0.1))
            if self.evalmode:
                self.data = self.data[start:start+250]
            else:
                self.data = self.data[:end] # + self.data[end:]
        elif split == 0.9:
            start = int(len(self.data) * split)
            # end = int(len(self.data) * (split + 0.05))
            end = start + 250
            if self.evalmode:
                self.data = self.data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def preprocessing(self, data):
        datasamples = []
        labels = []
        if self.mode == "gt" or "compression" in self.mode or "pewcrowd" in self.mode:
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
            if "bt" in self.mode:
                input_str1 = "Query: {}\nResponse:{}".format(data["query"], data["response1"])
                input_str2 = "Query: {}\nResponse:{}".format(data["query"], data["response2"])
            else:
                input_str = "Query: {}\n{}\nIs answer A better than answer B?".format(data["query"], data["response"])
        elif self.task == "mmlujudge":
            input_str = "Question:\n{}\nAnswer:\n{}\nIs the Answer to the Question correct?".format(data["query"], data["response"])
        elif self.task == "crosscheck":
            input_str = "Passage: {}\nIs there any non-factual or hallucinated information in the passage?".format(data["query"])
        else:
            input_str = "N/A"
        if "bt" in self.mode:
            prompt_inputs_1 = self.tokenizer(input_str1, return_tensors="pt")["input_ids"][0]
            prompt_inputs_2 = self.tokenizer(input_str2, return_tensors="pt")["input_ids"][0]
            prompt_inputs = [prompt_inputs_1, prompt_inputs_2]
        else:
            prompt_inputs = self.tokenizer(input_str, return_tensors="pt")["input_ids"][0]
        return prompt_inputs, torch.tensor(datasamples), torch.tensor(labels)


def collate_fn(batch):
    input_ids, workers, labels = zip(*batch)
    if len(input_ids[0]) > 1:
        input_ids = [x for xs in input_ids for x in xs]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    attn_mask = input_ids != 0
    attn_mask[:, 0] = True
    inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
    workers = torch.stack(workers).to(device)
    labels = torch.stack(labels).to(device)
    return inputs, workers, labels


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        tokenizer,
        multiturn="single",
    ):
        super(SupervisedDataset, self).__init__()
        self.data = []
        self.multiturn = multiturn
        with open(data_path) as fin:
            self.data = json.load(fin)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def preprocessing(self, data):
        messages = data[:2]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        )[0]
        return input_ids

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

def collate_sft_fn(batch):
    total_ids = pad_sequence(batch, batch_first=True, padding_value=0) #.to(device)
    print(total_ids.size())
    total_label = pad_sequence(batch, batch_first=True, padding_value=-1) #.to(device)
    attn_mask = total_ids != 0
    inputs = {"input_ids": total_ids, "attention_mask": attn_mask}
    return inputs, total_label