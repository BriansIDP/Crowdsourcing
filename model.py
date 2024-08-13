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
        num_layers=1,
    ):
        super(WorkerPredictor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir="/scratch/NeurowaveEval/leaderboard/bot/cache",  # Change to your local directory
        )
        # if mode != "gt":
        #     for name, param in self.llm.named_parameters():
        #         param.requires_grad = False
        self.nllms = nllms
        self.mode = mode
        inner_dim = self.llm.config.hidden_size + self.nllms - 1
        self.inner_dim = inner_dim + 1 if mode != "pew" else inner_dim
        self.outer_dim = self.inner_dim
        pos_emb_dim = 64
        if self.mode == "pew":
            for i in range(self.nllms):
                setattr(self, "outproj_{}".format(i+1), torch.nn.Linear(self.inner_dim, self.outer_dim))
                setattr(self, "outlayer_{}".format(i+1), torch.nn.Linear(self.outer_dim, 1))
        elif self.mode == "transformer":
            self.pos_emb = torch.nn.Embedding(self.nllms, pos_emb_dim)
            # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=1+pos_emb_dim+self.llm.config.hidden_size, nhead=1, batch_first=True)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=pos_emb_dim*2, nhead=1, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # self.output_mean = torch.nn.Linear(1+pos_emb_dim+self.llm.config.hidden_size, 1)
            # self.output_logdev = torch.nn.Linear(1+pos_emb_dim+self.llm.config.hidden_size, 1)
            self.output_mean = torch.nn.Linear(pos_emb_dim*2, 1)
            self.output_logdev = torch.nn.Linear(pos_emb_dim*2, 1)
        elif self.mode == "pewcrowd":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, 1 * self.nllms, bias=False)
            # self.outlayer.weight.data = torch.eye(2).unsqueeze(0).repeat(self.nllms, 1, 1).view(2 * self.nllms, 2)
            self.outlayer.weight.data = torch.cat((torch.ones(self.nllms, 1), 0*torch.ones(self.nllms, 1)), dim=-1)
        elif self.mode == "gt":
            self.output_layer = torch.nn.Linear(self.llm.config.hidden_size+self.nllms, 2)
        else:
            # self.kproj = torch.nn.Linear(self.llm.config.hidden_size+1+pos_emb_dim, 64)
            self.qproj = torch.nn.Linear(self.llm.config.hidden_size+1+pos_emb_dim, 64)
            self.kproj = torch.nn.Linear(1+pos_emb_dim, 64)
            self.z_hat_proj = torch.nn.Linear(self.outer_dim, 1)
            self.pos_emb = torch.nn.Embedding(self.nllms, pos_emb_dim)
        self.activation = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.1)
        self.tokenizer = tokenizer
        self.regression = regression

    def forward(self, inputs, workers, labels):
        if self.regression == "mse" and self.mode not in ["gt", "pewcrowd"]:
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
                # each_pred = self.drop(self.activation(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i])))
                each_pred = getattr(self, "outlayer_{}".format(i+1))(pred_hidden[:, i])
                pred_hiddens.append(each_pred)
            pred_hidden = torch.cat(pred_hiddens, dim=1)
            if self.regression == "logistic":
                pred_hidden = torch.sigmoid(pred_hidden)
            loss = ((pred_hidden - labels) ** 2).mean()
        elif self.mode == "pewcrowd":
            if self.regression == "hardlabel":
                labels = (labels < 0.5).long()
                group_labels = labels.sum(dim=-1) > self.nllms/2
            pred_hidden = self.bottleneck(self.drop(pred_hidden))
            # group_loss = torch.nn.functional.cross_entropy(pred_hidden, group_labels.long().view(-1))
            pred_hidden = torch.softmax(pred_hidden, dim=-1)
            # pred_hidden = torch.sigmoid(pred_hidden)
            # pred_hidden = self.outlayer(pred_hidden) #.view(pred_hidden.size(0)*self.nllms, 2)
            normalised_weight = torch.softmax(self.outlayer.weight, -1)
            pred_hidden = (pred_hidden.unsqueeze(1) * normalised_weight.unsqueeze(0)).sum(dim=-1)
            if self.regression == "skill":
                loss = ((pred_hidden.view(-1) - labels.view(-1)) ** 2).mean()
                # loss = - labels * torch.log(pred_hidden) - (1 - labels) * torch.log(1 - pred_hidden)
                # loss = loss.mean()
            else:
                pred_hidden = torch.log(torch.cat([pred_hidden.unsqueeze(-1), 1-pred_hidden.unsqueeze(-1)], dim=-1))
                loss = torch.nn.functional.cross_entropy(pred_hidden.view(pred_hidden.size(0)*self.nllms, 2), labels.view(-1))
            # loss = loss + 0 * group_loss
        elif self.mode == "transformer":
            # masking workers and get labels
            workers_in = workers.unsqueeze(-1).repeat(1, self.nllms, 1)
            input_mask_diag = torch.eye(self.nllms).unsqueeze(0).repeat(workers.size(0), 1, 1).to(workers.device)
            input_mask_diag = input_mask_diag.view(-1, self.nllms)
            # input_mask = torch.rand(workers.size(0)*workers.size(1), workers.size(1)).to(workers.device) > random.random()
            # input_mask = 1 - (input_mask * (1 - input_mask_diag))
            workers_in = workers_in.view(-1, self.nllms)
            workers_in = workers_in.masked_fill(input_mask_diag.bool(), 0).unsqueeze(-1)

            pos_inds = torch.tensor([i for i in range(self.nllms)]).to(workers.device)
            pos_embs = self.pos_emb(pos_inds).unsqueeze(0).repeat(workers_in.size(0), 1, 1)
            # pred_hidden = pred_hidden.unsqueeze(1).unsqueeze(1).repeat(1, self.nllms, self.nllms, 1).view(-1, self.nllms, pred_hidden.size(-1))
            # workers_in = torch.cat([workers_in, pos_embs, pred_hidden], dim=-1)
            # workers_in = torch.cat([workers_in, pos_embs], dim=-1)
            workers_in = torch.cat([workers_in * pos_embs, pos_embs], dim=-1)
            enc_out = self.transformer_encoder(workers_in)
            output_mean = self.output_mean(enc_out).squeeze(-1)
            output_mean = torch.diagonal(output_mean.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
            output_logdev = self.output_logdev(enc_out).squeeze(-1)
            output_logdev = torch.diagonal(output_logdev.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
            # loss = output_logdev + 0.5 * (output_mean - labels) ** 2 / (torch.exp(output_logdev) ** 2)
            loss = output_logdev + 0.5 * (labels) ** 2 / (torch.exp(output_logdev) ** 2)
            # loss = (output_mean - labels) ** 2
            loss = loss.mean()
        elif self.mode == "gt":
            pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
            prediction = self.output_layer(pred_hidden)
            loss = torch.nn.functional.cross_entropy(prediction, labels.view(-1))
        else:
            # Get inputs and masked inputs
            pred_hidden = pred_hidden.unsqueeze(1).repeat(1, self.nllms, 1)
            masked_pred_hidden = torch.cat([pred_hidden, workers.unsqueeze(-1)*0], dim=-1)
            # pred_hidden = torch.cat([pred_hidden, workers.unsqueeze(-1)], dim=-1)
            # masked_pred_hidden = workers.unsqueeze(-1)*0
            pred_hidden = workers.unsqueeze(-1)

            # Get positional embedding (learned)
            pos_inds = torch.tensor([i for i in range(self.nllms)]).to(workers.device)
            pos_embs = self.pos_emb(pos_inds).unsqueeze(0).repeat(pred_hidden.size(0), 1, 1)
            # pos_embs = pos_inds.unsqueeze(0).unsqueeze(-1).repeat(pred_hidden.size(0), 1, 1)

            # Append pos emb to the inputs
            masked_pred_hidden = torch.cat([masked_pred_hidden, pos_embs], dim=-1)
            pred_hidden = torch.cat([pred_hidden, pos_embs], dim=-1)

            # Forward QKV
            q_vecs = self.drop(self.qproj(masked_pred_hidden))
            k_vecs = self.drop(self.kproj(pred_hidden))
            # v_proj = self.drop(self.vproj(pred_hidden))

            # Compute attention
            modulus = torch.sqrt((q_vecs ** 2).sum(dim=-1)) * torch.sqrt((k_vecs ** 2).sum(dim=-1))
            scores = torch.einsum("bik,bjk->bij", q_vecs, k_vecs) / modulus.unsqueeze(-1)
            # scores = torch.sigmoid(scores)
            diag_mask = torch.eye(self.nllms).unsqueeze(0).repeat(scores.size(0), 1, 1).to(scores.device)
            if self.regression != "mse":
                scores = torch.softmax(scores.masked_fill(diag_mask.bool(), -1e9), dim=-1)
            else:
                scores = scores.masked_fill(diag_mask.bool(), 0)
            # scores = torch.softmax(scores, dim=-1)
            out_vecs = torch.einsum("bij,bjk->bik", scores, workers.unsqueeze(-1))
            loss = ((out_vecs.squeeze(-1) - labels) ** 2).mean()

        return loss

    def predict(self, inputs, workers, aggregation="mean", expected_error=1, labels=0):
        if self.regression == "mse" and self.mode != "gt":
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
                # each_pred = self.activation(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i]))
                each_pred = getattr(self, "outlayer_{}".format(i+1))(pred_hidden[:, i])
                pred_hiddens.append(each_pred)
            pred_hidden = torch.cat(pred_hiddens, dim=1)
            if self.regression == "logistic":
                pred_hidden = torch.sigmoid(pred_hidden)
            if aggregation == "mean":
                pred_hidden = pred_hidden.mean(dim=-1)
                prediction = pred_hidden > 0 if self.regression != "logistic" else pred_hidden > 0.5
            elif aggregation == "grad":
                pred_hidden.sum().backward()
                grad = workers.grad
                expected_p = torch.sigmoid(pred_hidden)
                expected_error = expected_p * (1 - expected_p)
                weight = []
                for k, row in enumerate(grad[0]):
                    weight.append(torch.cat((row[:k], grad.new_zeros(1), row[k:]), dim=-1))
                weight = torch.stack(weight, dim=0).sum(dim=0)
                # grad = grad / grad.sum(dim=-1, keepdim=True)
                # weight = (1 - weight).unsqueeze(0).repeat(workers.size(0), 1) / expected_error / 10
                weight = weight.unsqueeze(0).repeat(workers.size(0), 1) / expected_error / 10
                # Normalise weight
                weight = torch.softmax(weight, dim=-1)
                all_workers = torch.cat([workers[:, 1, 0:1], workers[:, 0]], dim=-1)
                pred_hidden = (all_workers * weight).sum(dim=-1)
                prediction = pred_hidden > 0 if self.regression != "logistic" else pred_hidden > 0.5
            elif aggregation == "ex_error":
                pred_hidden = torch.sigmoid(pred_hidden)
                prediction = pred_hidden * (1 - pred_hidden)
        elif self.mode == "pewcrowd":
            pred_hidden = self.bottleneck(pred_hidden)
            prediction = torch.softmax(pred_hidden, dim=-1)
            # prediction = torch.sigmoid(pred_hidden)
            # pred_hidden = self.outlayer(prediction).view(prediction.size(0)*self.nllms, 1)
            normalised_weight = torch.softmax(self.outlayer.weight.data, dim=-1)
            pred_hidden = (prediction.unsqueeze(1) * normalised_weight.unsqueeze(0)).sum(dim=-1).mean(dim=1).view(-1, 1)
            pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)

            # EM
            sigma = prediction[:, 0]
            p_r_0 = normalised_weight[:, 0]
            p_r_1 = 1 - normalised_weight[:, 1]
            numerator = torch.log(p_r_0).unsqueeze(0) * (1 - labels) + torch.log(1 - p_r_0) * labels
            numerator = sigma * torch.exp(numerator.sum(dim=-1))
            denominator = torch.log(p_r_1).unsqueeze(0) * labels + torch.log(1 - p_r_1) * (1 - labels)
            denominator = (1 - sigma) * torch.exp(denominator.sum(dim=-1))
            prediction = (numerator < denominator).float().unsqueeze(-1)
            prediction = torch.cat([1-prediction, prediction], dim=-1)
        elif self.mode == "gt":
            pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
            prediction = torch.softmax(self.output_layer(pred_hidden), dim=-1)
        elif self.mode == "transformer":
            # masking workers and get labels
            workers_in = workers.unsqueeze(-1).repeat(1, self.nllms, 1)
            input_mask_diag = torch.eye(self.nllms).unsqueeze(0).repeat(workers.size(0), 1, 1).to(workers.device)
            input_mask_diag = input_mask_diag.view(-1, self.nllms)
            # input_mask = torch.random(workers.size(0), workers.size(1), workers.size(0)).to(workers.device) > random.random()
            # input_mask = 1 - (input_mask * (1 - input_mask_diag))
            workers_in = workers_in.view(-1, self.nllms)
            workers_in = workers_in.masked_fill(input_mask_diag.bool(), 0).unsqueeze(-1)

            pos_inds = torch.tensor([i for i in range(self.nllms)]).to(workers.device)
            # pos_embs = pos_inds.unsqueeze(0).unsqueeze(-1).repeat(workers_in.size(0), 1, 1)
            pos_embs = self.pos_emb(pos_inds).unsqueeze(0).repeat(workers_in.size(0), 1, 1)
            # pred_hidden = pred_hidden.unsqueeze(1).unsqueeze(1).repeat(1, self.nllms, self.nllms, 1).view(-1, self.nllms, pred_hidden.size(-1))
            # workers_in = torch.cat([workers_in, pos_embs, pred_hidden], dim=-1)
            # workers_in = torch.cat([workers_in, pos_embs], dim=-1)
            workers_in = torch.cat([workers_in * pos_embs, pos_embs], dim=-1)
            enc_out = self.transformer_encoder(workers_in)
            output_mean = self.output_mean(enc_out).squeeze(-1)
            output_mean = torch.diagonal(output_mean.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
            output_logdev = self.output_logdev(enc_out).squeeze(-1)
            output_logdev = torch.diagonal(output_logdev.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
            prediction = output_mean.mean(dim=-1) > 0
        else:
            # Get inputs and masked inputs
            pred_hidden = pred_hidden.unsqueeze(1).repeat(1, self.nllms, 1)
            masked_pred_hidden = torch.cat([pred_hidden, workers.unsqueeze(-1)*0], dim=-1)
            # pred_hidden = torch.cat([pred_hidden, workers.unsqueeze(-1)], dim=-1)
            # masked_pred_hidden = workers.unsqueeze(-1)*0
            pred_hidden = workers.unsqueeze(-1)

            # Get positional embedding (learned)
            pos_inds = torch.tensor([i for i in range(self.nllms)]).to(workers.device)
            pos_embs = self.pos_emb(pos_inds).unsqueeze(0).repeat(pred_hidden.size(0), 1, 1)
            # pos_embs = pos_inds.unsqueeze(0).unsqueeze(-1).repeat(pred_hidden.size(0), 1, 1)

            # Append pos emb to the inputs
            masked_pred_hidden = torch.cat([masked_pred_hidden, pos_embs], dim=-1)
            pred_hidden = torch.cat([pred_hidden, pos_embs], dim=-1)

            # Forward QKV
            q_vecs = self.drop(self.qproj(masked_pred_hidden))
            k_vecs = self.drop(self.kproj(pred_hidden))
            # v_proj = self.drop(self.vproj(pred_hidden))

            # Compute attention
            modulus = torch.sqrt((q_vecs ** 2).sum(dim=-1)) * torch.sqrt((k_vecs ** 2).sum(dim=-1))
            scores = torch.einsum("bik,bjk->bij", q_vecs, k_vecs) / modulus.unsqueeze(-1) # math.sqrt(q_vecs.size(-1))
            # scores = torch.sigmoid(scores)
            diag_mask = torch.eye(self.nllms).unsqueeze(0).repeat(scores.size(0), 1, 1).to(scores.device)
            if self.regression != "mse":
                scores = torch.softmax(scores.masked_fill(diag_mask.bool(), -1e9), dim=-1)
            else:
                scores = scores.masked_fill(diag_mask.bool(), 0)
            # scores = torch.softmax(scores, dim=-1)
            pred_hidden = torch.einsum("bij,bjk->bik", scores, workers.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
            prediction = pred_hidden > 0
        return prediction, pred_hidden

    def density_estimtion(self, inputs, workers):
        workers = - torch.log(1 / (workers) - 1)
        pos_inds = torch.tensor([i for i in range(self.nllms)]).to(workers.device)
        pos_embs = self.pos_emb(pos_inds).unsqueeze(0)
        workers_mask1 = (workers * torch.tensor([0, 1] + [1] * (self.nllms - 2)).to(workers.device)).unsqueeze(-1)
        workers_mask2 = (workers * torch.tensor([1, 0] + [1] * (self.nllms - 2)).to(workers.device)).unsqueeze(-1)
        workers_mask12 = (workers * torch.tensor([0, 0] + [1] * (self.nllms - 2)).to(workers.device)).unsqueeze(-1)
        # workers_in_1 = workers_mask1 * pos_embs
        # workers_in_2 = workers_mask2 * pos_embs
        # workers_in_12 = workers_mask12 * pos_embs
        workers_in_1 = torch.cat([workers_mask1 * pos_embs, pos_embs], dim=-1)
        workers_in_2 = torch.cat([workers_mask2 * pos_embs, pos_embs], dim=-1)
        workers_in_12 = torch.cat([workers_mask12 * pos_embs, pos_embs], dim=-1)
        workers_in = torch.cat([workers_in_1, workers_in_2, workers_in_12], dim=0)
        enc_out = self.transformer_encoder(workers_in)
        output_mean = self.output_mean(enc_out).squeeze(-1)
        # output_mean = torch.diagonal(output_mean.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
        output_logdev = self.output_logdev(enc_out).squeeze(-1)
        sigma_123 = torch.exp(output_logdev[0, 0]) ** 2
        sigma_213 = torch.exp(output_logdev[1, 1]) ** 2
        sigma_13 = torch.exp(output_logdev[2, 0]) ** 2
        sigma_23 = torch.exp(output_logdev[2, 1]) ** 2
        # output_logdev = torch.diagonal(output_logdev.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
        return [sigma_123.item(), sigma_213.item(), sigma_13.item(), sigma_23.item()]
