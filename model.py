import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json
from copy import deepcopy

import numpy as np
import six
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel


class WorkerPredictor(torch.nn.Module):
    def __init__(
        self,
        model_path,
        nllms,
        tokenizer,
        regression="mse",
    ):
        super(WorkerPredictor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir="/scratch/NeurowaveEval/leaderboard/bot/cache",
        )
        self.nllms = nllms
        inner_dim = self.llm.config.hidden_size + self.nllms - 1
        for i in range(self.nllms):
            setattr(self, "outproj_{}".format(i+1), torch.nn.Linear(inner_dim, inner_dim))
            setattr(self, "outlayer_{}".format(i+1), torch.nn.Linear(inner_dim, 1))
        self.tokenizer = tokenizer
        self.regression = regression

    def forward(self, inputs, workers, labels):
        workers = - torch.log(1 / workers - 1 + 1e-5)
        # worker_mask = torch.eye(self.nllms).unsqueeze(0).to(workers.device)
        # workers *= (1 - worker_mask)
        if self.regression == "mse":
            labels = - torch.log(1 / labels - 1 + 1e-5)
        attention_mask = inputs["attention_mask"]
        outputs = self.llm(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        pred_hidden = pred_hidden.unsqueeze(1).repeat(1, self.nllms, 1)
        pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
        pred_hiddens = []
        for i in range(self.nllms):
            each_pred = torch.relu(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i]))
            each_pred = getattr(self, "outlayer_{}".format(i+1))(pred_hidden[:, i])
            pred_hiddens.append(each_pred)
        pred_hidden = torch.cat(pred_hiddens, dim=1)
        if self.regression == "mse":
            loss = ((pred_hidden - labels) ** 2).mean()
        elif self.regression == "logistic":
            pred_hidden = torch.sigmoid(pred_hidden)
            loss = - labels * torch.log(pred_hidden) - (1 - labels) * torch.log(1 - pred_hidden)
            loss = loss.mean()
        return loss

    def predict(self, inputs, workers, aggregation="mean", expected_error=1):
        if self.regression == "mse":
            workers = - torch.log(1 / workers - 1 + 1e-5)
        all_workers = torch.cat([workers[:, 1, 0:1], workers[:, 0]], dim=-1)
        # worker_mask = torch.eye(self.nllms).unsqueeze(0).to(workers.device)
        # workers *= (1 - worker_mask)
        workers.requires_grad = True
        attention_mask = inputs["attention_mask"]
        outputs = self.llm(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        pred_hidden = pred_hidden.unsqueeze(1).repeat(1, self.nllms, 1)
        pred_hidden = torch.cat([pred_hidden, workers], dim=-1)

        pred_hiddens = []
        for i in range(self.nllms):
            each_pred = torch.relu(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i]))
            each_pred = getattr(self, "outlayer_{}".format(i+1))(pred_hidden[:, i])
            pred_hiddens.append(each_pred)
        pred_hidden = torch.cat(pred_hiddens, dim=1)
        if self.regression == "logistic":
            pred_hidden = torch.sigmoid(pred_hidden)
        if aggregation == "mean":
            pred_hidden = pred_hidden.mean(dim=-1)
            prediction = pred_hidden > 0 if self.regression != "logistic" else pred_hidden > 0.5
        elif aggregation == "grad":
            pred_hidden = pred_hidden.sum()
            pred_hidden.backward()
            grad = workers.grad.sum(dim=-1)
            grad = grad / grad.sum(dim=-1, keepdim=True)
            weight = (1 - grad) * expected_error
            # Normalise weight
            weight = torch.softmax(weight, dim=-1)
            prediction = (all_workers * weight).sum(dim=-1)
            prediction = prediction > 0 if self.regression != "logistic" else prediction > 0.5
        elif aggregation == "ex_error":
            prediction = (pred_hidden - all_workers) ** 2
        return prediction
