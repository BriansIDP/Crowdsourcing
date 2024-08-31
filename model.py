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
# import six
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel


class WorkerCompressor(torch.nn.Module):
    def __init__(
        self,
        nllms,
        target_llms,
        kl_factor=0.0,
    ):
        super(WorkerCompressor, self).__init__()
        self.nllms = nllms
        self.comp_llms = target_llms
        self.encoder = torch.nn.Linear(nllms, target_llms, bias=False)
        self.decoder = torch.nn.Linear(target_llms, nllms, bias=False)
        self.encoder.weight.data = 1/ nllms * torch.ones(target_llms, nllms)
        self.kl_factor = kl_factor
        self.epoch = 0

    def forward(self, workers, evalmode=False):
        normalised_weights = torch.softmax(self.encoder.weight, dim=-1)
        compressed = torch.einsum('bi,ji->bj', workers, normalised_weights)
        # compressed = torch.sigmoid(self.encoder(workers))
        # compressed = self.encoder(workers)
        # recovered = torch.sigmoid(self.decoder(compressed))
        recovered = self.decoder(compressed)
        loss = ((recovered - workers) ** 2).mean()
        # if evalmode:
        #     loss = (recovered - workers).abs().mean()
        # else:
        #     loss = - workers * torch.log(recovered) - (1 - workers) * torch.log(1 - recovered)
        #     loss = loss.mean()
        return loss, compressed


class WorkerPredictor(torch.nn.Module):
    def __init__(
        self,
        model_path,
        nllms,
        tokenizer,
        regression="mse",
        mode="pew",
        num_layers=1,
        lora_config={},
        reg_factor=0,
        freeze_epoch=0,
    ):
        super(WorkerPredictor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            # cache_dir="/scratch/NeurowaveEval/leaderboard/bot/cache",  # Change to your local directory
            cache_dir="scratch/cache",  # Change to your local directory
        )
        if model_path != "gpt2":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config["lora_rank"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                target_modules=lora_config["lora_module"],
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
        self.nllms = nllms
        self.mode = mode
        inner_dim = self.llm.config.hidden_size + self.nllms - 1
        self.inner_dim = inner_dim + 1 if mode != "pew" else inner_dim
        self.outer_dim = self.inner_dim
        pos_emb_dim = self.nllms
        if self.mode == "pew":
            # add a hidden layer for each evidence llm
            for i in range(self.nllms):
                setattr(self, "outproj_{}".format(i+1), torch.nn.Linear(self.inner_dim, self.outer_dim))
                setattr(self, "outlayer_{}".format(i+1), torch.nn.Linear(self.outer_dim, 1))
        elif self.mode == "transformer":
            self.pos_emb = torch.nn.Embedding(self.nllms, pos_emb_dim)
            # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=1+pos_emb_dim+self.llm.config.hidden_size, nhead=1, batch_first=True)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=pos_emb_dim+1, nhead=1, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_mean = torch.nn.Linear(pos_emb_dim+1, 1)
            self.output_logdev = torch.nn.Linear(pos_emb_dim+1, 1)
        elif self.mode == "pewcrowd":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, 2 * self.nllms, bias=False)
            self.outlayer.weight.data = torch.eye(2).unsqueeze(0).repeat(self.nllms, 1, 1).view(2 * self.nllms, 2)
        elif self.mode == "pewcrowdimp" or self.mode == "pewcrowdae":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, self.nllms, bias=False)
            self.outlayer.weight.data = torch.cat((0.7 * torch.ones(self.nllms, 1), 0.3 * torch.ones(self.nllms, 1)), dim=-1)
        elif self.mode == "pewcrowdimpxt":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, self.nllms, bias=False)
            self.outlayer.weight.data = torch.cat((0.7 * torch.ones(self.nllms, 1), 0.3 * torch.ones(self.nllms, 1)), dim=-1)
            self.skilllayer = torch.nn.Linear(self.llm.config.hidden_size, 2 * self.nllms)
        elif self.mode == "pewcrowdaepost":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, self.nllms, bias=False)
            self.outlayer.weight.data = torch.cat((0.7 * torch.ones(self.nllms, 1), 0.3 * torch.ones(self.nllms, 1)), dim=-1)
            self.correlationlayer = torch.nn.Linear(self.nllms-1, self.nllms, bias=False)
            self.correlationlayer.weight.data = torch.eye(self.nllms, self.nllms)
        elif self.mode == "gt":
            self.output_layer = torch.nn.Linear(self.llm.config.hidden_size, 2)
        else:
            # self.kproj = torch.nn.Linear(self.llm.config.hidden_size+1+pos_emb_dim, 64)
            self.qproj = torch.nn.Linear(self.llm.config.hidden_size+1+pos_emb_dim, 64)
            self.kproj = torch.nn.Linear(1+pos_emb_dim, 64)
            self.z_hat_proj = torch.nn.Linear(self.outer_dim, 1)
            # self.pos_emb = torch.nn.Embedding(self.nllms, pos_emb_dim)
        self.activation = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.1)
        self.tokenizer = tokenizer
        self.regression = regression
        self.regularisation = reg_factor
        self.epoch = 0
        self.freeze_epoch = freeze_epoch

    def freeze_model(self):
        for name, param in self.llm.named_parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for name, param in self.llm.named_parameters():
            param.requires_grad = True

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
                # each_pred = self.drop(self.activation(getattr(self, "outproj_{}".format(i+1))(pred_hidden[:, i])))
                each_pred = getattr(self, "outlayer_{}".format(i+1))(pred_hidden[:, i])
                pred_hiddens.append(each_pred)
            pred_hidden = torch.cat(pred_hiddens, dim=1)
            if self.regression == "logistic":
                pred_hidden = torch.sigmoid(pred_hidden)
            loss = ((pred_hidden - labels) ** 2).mean()
        elif self.mode == "pewcrowd":
            if self.regression == "hardlabel":
                labels = (workers < 0.5).long()
            latent_dist = self.bottleneck(self.drop(pred_hidden))
            latent_dist = torch.softmax(latent_dist, dim=-1)
            pred_hidden = self.outlayer(latent_dist).view(pred_hidden.size(0)*self.nllms, 2)
            if self.regression == "skill":
                # loss = ((pred_hidden.view(-1) - labels.view(-1)) ** 2).mean()
                pred_hidden = torch.softmax(pred_hidden, dim=-1).view(workers.size(0)*self.nllms, 2)
                workers = workers.view(-1)
                loss = - workers * torch.log(pred_hidden[:, 0]) - (1 - workers) * torch.log(pred_hidden[:, 1])
                loss = loss.mean()
            else:
                # pred_hidden = torch.log(torch.cat([pred_hidden.unsqueeze(-1), 1-pred_hidden.unsqueeze(-1)], dim=-1))
                loss = torch.nn.functional.cross_entropy(pred_hidden.view(labels.size(0)*self.nllms, 2), labels.view(-1))
        elif self.mode == "pewcrowdimp" or self.mode == "pewcrowdae":
            if self.regression == "hardlabel":
                labels = (workers < 0.5).long()
            latent_dist = self.bottleneck(self.drop(pred_hidden))
            latent_dist = torch.softmax(latent_dist, dim=-1)
            pred_hidden = self.outlayer(latent_dist)
            # normalised_weight = torch.softmax(self.outlayer.weight, -1).unsqueeze(0)
            # pred_hidden = (latent_dist.unsqueeze(1) * normalised_weight).sum(dim=-1)
            if self.regression == "skill":
                # loss = ((pred_hidden.view(-1) - labels.view(-1)) ** 2).mean()
                loss = - workers * torch.log(pred_hidden) - (1 - workers) * torch.log(1 - pred_hidden)
                loss = loss.mean()
                loss += self.regularisation * ((self.outlayer.weight[:,0] - self.outlayer.weight[:,1]) ** 2).mean()
            else:
                pred_hidden = torch.log(torch.cat([pred_hidden.unsqueeze(-1), 1-pred_hidden.unsqueeze(-1)], dim=-1))
                loss = torch.nn.functional.cross_entropy(pred_hidden.view(labels.size(0)*self.nllms, 2), labels.view(-1))
        elif self.mode == "pewcrowdimpxt":
            if self.regression == "hardlabel":
                labels = (workers < 0.5).long()
            latent_dist = self.bottleneck(self.drop(pred_hidden))
            latent_dist = torch.softmax(latent_dist, dim=-1)
            normalised_weight = self.skilllayer(pred_hidden.detach())
            normalised_weight = torch.sigmoid(normalised_weight).view(pred_hidden.size(0), self.nllms, 2)
            if self.epoch < self.freeze_epoch:
                pred_hidden = self.outlayer(latent_dist)
                extra_loss = ((normalised_weight - self.outlayer.weight.data) ** 2).mean()
                normalised_weight = self.outlayer.weight.unsqueeze(0)
            else:
                pred_hidden = (normalised_weight * latent_dist.detach().unsqueeze(1)).sum(dim=-1)
                extra_loss = 0
            if self.regression == "skill":
                # loss = ((pred_hidden.view(-1) - labels.view(-1)) ** 2).mean()
                loss = - workers * torch.log(pred_hidden) - (1 - workers) * torch.log(1 - pred_hidden)
                loss = loss.mean()
                loss += self.regularisation * ((normalised_weight[:, :, 0] - normalised_weight[:, :, 1]) ** 2).mean()
                loss += extra_loss
            else:
                pred_hidden = torch.log(torch.cat([pred_hidden.unsqueeze(-1), 1-pred_hidden.unsqueeze(-1)], dim=-1))
                loss = torch.nn.functional.cross_entropy(pred_hidden.view(labels.size(0)*self.nllms, 2), labels.view(-1))
        elif self.mode == "pewcrowdaepost":
            if self.regression == "hardlabel":
                labels = (workers < 0.5).long()
            latent_dist = self.bottleneck(self.drop(pred_hidden))
            latent_dist = torch.softmax(latent_dist, dim=-1)
            pred_hidden = self.outlayer(latent_dist)
            # normalised_weight = torch.softmax(self.outlayer.weight, -1).unsqueeze(0)
            # pred_hidden = (latent_dist.unsqueeze(1) * normalised_weight).sum(dim=-1)
            # pred_hidden = torch.sigmoid(self.correlationlayer(pred_hidden))
            if self.regression == "skill":
                # loss = ((pred_hidden.view(-1) - labels.view(-1)) ** 2).mean()
                loss = - workers * torch.log(pred_hidden) - (1 - workers) * torch.log(1 - pred_hidden)
                loss = loss.mean()
                loss += self.regularisation * ((self.outlayer.weight[:,0] - self.outlayer.weight[:,1]) ** 2).mean()
            else:
                pred_hidden = torch.log(torch.cat([pred_hidden.unsqueeze(-1), 1-pred_hidden.unsqueeze(-1)], dim=-1))
                loss = torch.nn.functional.cross_entropy(pred_hidden.view(labels.size(0)*self.nllms, 2), labels.view(-1))
            
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
            # pos_embs = self.pos_emb(pos_inds).unsqueeze(0).repeat(workers_in.size(0), 1, 1)
            pos_embs = torch.nn.functional.one_hot(torch.arange(0, self.nllms)).to(workers.device).unsqueeze(0).repeat(workers_in.size(0), 1, 1)
            workers_in = torch.cat([workers_in, pos_embs], dim=-1)
            enc_out = self.transformer_encoder(workers_in)
            output_mean = self.output_mean(enc_out).squeeze(-1)
            output_mean = torch.diagonal(output_mean.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
            output_logdev = self.output_logdev(enc_out).squeeze(-1)
            output_logdev = torch.diagonal(output_logdev.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
            # loss = output_logdev + 0.5 * (output_mean - labels) ** 2 / (torch.exp(output_logdev) ** 2)
            # loss = output_logdev + 0.5 * (labels) ** 2 / (torch.exp(output_logdev) ** 2)
            loss = (output_mean - labels) ** 2
            loss = loss.mean()
        elif self.mode == "gt":
            prediction = self.output_layer(pred_hidden)
            loss = torch.nn.functional.cross_entropy(prediction, labels)
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

    def predict(self, inputs, workers, aggregation="mean", expected_error=1, labels=0, withEM=False):
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
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            pred_hidden = 1 - torch.softmax(self.outlayer(prediction).view(prediction.size(0)*self.nllms, 2), dim=-1)
        elif self.mode == "pewcrowdimp" or self.mode == "pewcrowdae":
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            pred_hidden = self.outlayer(prediction).view(prediction.size(0)*self.nllms, 1)
            normalised_weight = self.outlayer.weight.data
            # normalised_weight = torch.softmax(self.outlayer.weight.data, dim=-1).unsqueeze(0)
            # pred_hidden = (prediction.unsqueeze(1) * normalised_weight).sum(dim=-1).view(-1, 1)
            pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)

            # EM E-step
            if withEM:
                sigma = prediction[:, 0]
                p_r_0 = normalised_weight[:, 0]
                p_r_1 = 1 - normalised_weight[:, 1]
                numerator = torch.log(p_r_0).unsqueeze(0) * (1 - workers) + torch.log(1 - p_r_0) * workers
                numerator = sigma * torch.exp(numerator.sum(dim=-1))
                denominator = torch.log(p_r_1).unsqueeze(0) * workers + torch.log(1 - p_r_1) * (1 - workers)
                denominator = (1 - sigma) * torch.exp(denominator.sum(dim=-1))
                prediction = (numerator < denominator).float().unsqueeze(-1)
                prediction = torch.cat([1-prediction, prediction], dim=-1)
        elif self.mode == "pewcrowdimpxt":
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            normalised_weight = self.skilllayer(pred_hidden)
            normalised_weight = torch.sigmoid(normalised_weight).view(pred_hidden.size(0), self.nllms, 2)
            # normalised_weight = self.outlayer.weight.data.unsqueeze(0)
            if self.epoch < self.freeze_epoch:
                pred_hidden = self.outlayer(prediction).view(prediction.size(0)*self.nllms, 1)
            else:
                pred_hidden = (normalised_weight * prediction.unsqueeze(1)).sum(dim=-1).view(prediction.size(0)*self.nllms, 1)
            pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)

            # EM E-step
            if withEM:
                sigma = prediction[:, 0]
                p_r_0 = normalised_weight[:, :, 0]
                p_r_1 = 1 - normalised_weight[:, :, 1]
                numerator = torch.log(p_r_0) * (1 - labels) + torch.log(1 - p_r_0) * labels
                numerator = sigma * torch.exp(numerator.sum(dim=-1))
                denominator = torch.log(p_r_1) * labels + torch.log(1 - p_r_1) * (1 - labels)
                denominator = (1 - sigma) * torch.exp(denominator.sum(dim=-1))
                prediction = (numerator < denominator).float().unsqueeze(-1)
                prediction = torch.cat([1-prediction, prediction], dim=-1)
        elif self.mode == "pewcrowdaepost":
            if self.mode == "pewcrowdaepost":
                pred_hidden = torch.cat([pred_hidden, labels], dim=-1)
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            pred_hidden = self.outlayer(prediction).view(prediction.size(0)*self.nllms, 1)
            # normalised_weight = torch.softmax(self.outlayer.weight, -1).unsqueeze(0)
            # pred_hidden = (latent_dist.unsqueeze(1) * normalised_weight).sum(dim=-1)
            # pred_hidden = torch.sigmoid(self.correlationlayer(pred_hidden)).view(prediction.size(0)*self.nllms, 1)
            pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)

        elif self.mode == "gt":
            # pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
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
        # pos_embs = self.pos_emb(pos_inds).unsqueeze(0)
        pos_embs = torch.nn.functional.one_hot(torch.arange(0, self.nllms)).to(workers.device).unsqueeze(0)
        workers_mask1 = (workers * torch.tensor([0, 1] + [1] * (self.nllms - 2)).to(workers.device)).unsqueeze(-1)
        workers_mask2 = (workers * torch.tensor([1, 0] + [1] * (self.nllms - 2)).to(workers.device)).unsqueeze(-1)
        workers_mask12 = (workers * torch.tensor([0, 0] + [1] * (self.nllms - 2)).to(workers.device)).unsqueeze(-1)
        # workers_in_1 = workers_mask1 * pos_embs
        # workers_in_2 = workers_mask2 * pos_embs
        # workers_in_12 = workers_mask12 * pos_embs
        workers_in_1 = torch.cat([workers_mask1, pos_embs], dim=-1)
        workers_in_2 = torch.cat([workers_mask2, pos_embs], dim=-1)
        workers_in_12 = torch.cat([workers_mask12, pos_embs], dim=-1)
        workers_in = torch.cat([workers_in_1, workers_in_2, workers_in_12], dim=0)
        enc_out = self.transformer_encoder(workers_in)
        output_mean = self.output_mean(enc_out).squeeze(-1)
        # output_mean = torch.diagonal(output_mean.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
        output_logdev = self.output_logdev(enc_out).squeeze(-1)
        sigma_123 = output_mean[0,0] # torch.exp(output_logdev[0, 0]) ** 2
        sigma_213 = output_mean[1,1] # torch.exp(output_logdev[1, 1]) ** 2
        sigma_13 = output_mean[2,0] # torch.exp(output_logdev[2, 0]) ** 2
        sigma_23 = output_mean[2,1] #torch.exp(output_logdev[2, 1]) ** 2
        # output_logdev = torch.diagonal(output_logdev.view(workers.size(0), workers.size(1), -1), dim1=1, dim2=2)
        return [sigma_123.item(), sigma_213.item(), sigma_13.item(), sigma_23.item()]
