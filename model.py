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
        mode="pew",
    ):
        super(WorkerPredictor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir="/scratch/NeurowaveEval/leaderboard/bot/cache",  # Change to your local directory
        )
        self.nllms = nllms
        self.mode = mode
        inner_dim = self.llm.config.hidden_size + self.nllms - 1
        inner_dim = inner_dim + 1 if mode != "pew" else inner_dim
        outer_dim = inner_dim
        if self.mode == "pew":
            for i in range(self.nllms):
                setattr(self, "outproj_{}".format(i+1), torch.nn.Linear(inner_dim, outer_dim))
                setattr(self, "outlayer_{}".format(i+1), torch.nn.Linear(outer_dim, 1))
        else:
            self.outproj = torch.nn.Linear(inner_dim, outer_dim)
            self.z_hat_proj = torch.nn.Linear(outer_dim, 1)
            self.z_hat_scale = torch.nn.Linear(outer_dim, self.nllms)
            self.noise_variance = torch.nn.Linear(outer_dim, self.nllms*self.nllms)
        self.activation = torch.nn.ReLU()
        self.tokenizer = tokenizer
        self.regression = regression

    def forward(self, inputs, workers, labels):
        if self.regression == "mse":
            workers = - torch.log(1 / (workers) - 1)
            labels = - torch.log(1 / (labels) - 1)
        attention_mask = inputs["attention_mask"]
        outputs = self.llm(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        if self.mode == "pew":
            pred_hidden = pred_hidden.unsqueeze(1).repeat(1, self.nllms, 1)
            pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
            pred_hiddens = []
            for i in range(self.nllms):
                each_pred = self.activation(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i]))
                each_pred = getattr(self, "outlayer_{}".format(i+1))(each_pred)
                pred_hiddens.append(each_pred)
            pred_hidden = torch.cat(pred_hiddens, dim=1)
            if self.regression == "logistic":
                pred_hidden = torch.sigmoid(pred_hidden)
            loss = ((pred_hidden - labels) ** 2).mean()
        else:
            pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
            pred_hidden = self.activation(self.outproj(pred_hidden))
            z_hat_sign = self.z_hat_proj(pred_hidden)
            z_hat_scale = torch.sigmoid(self.z_hat_scale(pred_hidden)) * 10
            log_noise_std = self.noise_variance(pred_hidden).view(pred_hidden.size(0), self.nllms, self.nllms)
            noise_std = torch.exp(log_noise_std)
            noise_var = torch.einsum('bij,bjk->bik', noise_std.transpose(1, 2), noise_std) / math.sqrt(math.exp(self.nllms))
            z_hat_value = z_hat_sign * z_hat_scale
            # noise_var = torch.diag_embed(torch.exp(log_noise_var), offset=0, dim1=-2, dim2=-1)
            Y_unbiased = labels - z_hat_value
            noise_var_inv = torch.linalg.inv(noise_var)
            loss = torch.diag(torch.nn.functional.bilinear(Y_unbiased, Y_unbiased, noise_var_inv)) + torch.logdet(noise_var)
            loss = loss.mean()
        return loss

    def predict(self, inputs, workers, aggregation="mean", expected_error=1):
        if self.regression == "mse":
            workers = - torch.log(1 / (workers) - 1)
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

        if self.mode == "pew":
            pred_hidden = pred_hidden.unsqueeze(1).repeat(1, self.nllms, 1)
            pred_hidden = torch.cat([pred_hidden, workers], dim=-1)

            pred_hiddens = []
            for i in range(self.nllms):
                each_pred = self.activation(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i]))
                each_pred = getattr(self, "outlayer_{}".format(i+1))(each_pred)
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
                # grad = grad / grad.sum(dim=-1, keepdim=True)
                weight = (1 - grad) # / expected_error
                # Normalise weight
                weight = torch.softmax(weight, dim=-1)
                all_workers = torch.cat([workers[:, 1, 0:1], workers[:, 0]], dim=-1)
                pred_hidden = (all_workers * weight).sum(dim=-1)
                prediction = pred_hidden > 0 if self.regression != "logistic" else pred_hidden > 0.5
            elif aggregation == "ex_error":
                pred_hidden = torch.sigmoid(pred_hidden)
                prediction = pred_hidden * (1 - pred_hidden)
        else:
            pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
            pred_hidden = self.activation(self.outproj(pred_hidden))
            z_hat_scale = torch.sigmoid(self.z_hat_scale(pred_hidden)) * 10
            log_noise_std = self.noise_variance(pred_hidden).view(pred_hidden.size(0), self.nllms, self.nllms)
            noise_std = torch.exp(log_noise_std)
            noise_var = torch.einsum('bij,bjk->bik', noise_std.transpose(1, 2), noise_std) / math.sqrt(math.exp(self.nllms))
            pred_hidden = self.z_hat_proj(pred_hidden)[:, 0]
            prediction = pred_hidden > 0
        return prediction, pred_hidden
