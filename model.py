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
from transformers import AutoModelForCausalLM, MambaForCausalLM
from transformers import AutoModelForSeq2SeqLM, RobertaModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from scipy.stats import beta


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


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
        self.encoder = torch.nn.Linear(nllms+2, target_llms, bias=False)
        # self.encoder = torch.nn.Linear(nllms, target_llms, bias=False)
        self.decoder = torch.nn.Linear(target_llms, nllms)
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(target_llms, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, nllms),
        # )
        self.encoder.weight.data = torch.zeros(target_llms, nllms+2)
        # self.encoder.weight.data = torch.zeros(target_llms, nllms)
        self.kl_factor = kl_factor
        if self.kl_factor > 0:
            self.discriminator = NDM(target_llms)
        self.epoch = 0

    def forward(self, workers, evalmode=False):
        workers_aug = torch.cat([workers, workers.new_ones(workers.size(0), 1), workers.new_zeros(workers.size(0), 1)], dim=-1)
        normalised_weights = torch.softmax(self.encoder.weight, dim=-1)
        compressed = torch.einsum('bi,ji->bj', workers_aug, normalised_weights)
        recovered = torch.sigmoid(self.decoder(compressed))
        if evalmode:
            loss = (recovered - workers).abs().mean()
        else:
            loss = - workers * torch.log(recovered) - (1 - workers) * torch.log(1 - recovered)
            loss = loss.mean()
            if self.kl_factor > 0:
                kl_loss = self.compute_KL(compressed)
                loss = loss + self.kl_factor * kl_loss
        return loss, compressed

    def compute_KL(self, compressed):
        with torch.no_grad():
            priors = []
            for i in range(self.comp_llms):
                one_alpha_1, one_alpha_2, loc, scale = beta.fit(compressed[:, i].cpu().numpy(), floc=0, fscale=1)
                one_samples = beta.rvs(one_alpha_1, one_alpha_2, size=compressed.size(0))
                priors.append(torch.tensor(one_samples).to(compressed.device).view(-1, 1).float())
            prior_compressed = torch.cat(priors, dim=-1)
            # prior_compressed = torch.rand(compressed.size(0), self.comp_llms).to(compressed.device)
        compressed = grad_reverse(compressed)
        inputs = torch.cat([prior_compressed, compressed], dim=0)
        labels = torch.cat([torch.ones(compressed.size(0)), torch.zeros(compressed.size(0))], dim=0).long().to(compressed.device)
        loss = self.discriminator(inputs, labels)
        return loss


class NDM(torch.nn.Module):
    def __init__(
        self,
        nfeatures,
    ):
        super(NDM, self).__init__()
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(nfeatures, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, nfeatures),
        )
    
    def forward(self, inputs, labels):
        outputs = self.discriminator(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss.mean()


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
        beta_factor=1.0,
    ):
        super(WorkerPredictor, self).__init__()
        self.model_path = model_path
        if "roberta" in model_path:
            self.llm = RobertaModel.from_pretrained(
                "FacebookAI/roberta-base",
                cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",  # Change to your local directory
                torch_dtype=torch.float32,
            )
        if "llama" in model_path or "gemma" in model_path:
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
        if self.mode == "pewcrowd":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, 2 * self.nllms, bias=False)
            self.outlayer.weight.data = torch.eye(2).unsqueeze(0).repeat(self.nllms, 1, 1).view(2 * self.nllms, 2)
        elif self.mode == "pewcrowdimp" or self.mode == "pewcrowdae":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, self.nllms, bias=False)
            self.outlayer.weight.data = torch.cat((0.7 * torch.ones(self.nllms, 1), 0.3 * torch.ones(self.nllms, 1)), dim=-1)
        elif self.mode == "pewcrowdimpxt" or self.mode == "pewcrowdaext":
            self.bottleneck = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.outlayer = torch.nn.Linear(2, self.nllms, bias=False)
            self.outlayer.weight.data = torch.cat((0.7 * torch.ones(self.nllms, 1), 0.3 * torch.ones(self.nllms, 1)), dim=-1)
            self.skilllayer = torch.nn.Linear(self.llm.config.hidden_size, 2 * self.nllms)
        elif self.mode == "btcrowd":
            self.outlayer = torch.nn.Linear(2, 2 * self.nllms, bias=False)
            self.outlayer.weight.data = torch.eye(2).unsqueeze(0).repeat(self.nllms, 1, 1).view(2 * self.nllms, 2)
        elif self.mode == "btskillagg":
            self.outlayer = torch.nn.Linear(2, self.nllms, bias=False)
            self.outlayer.weight.data = torch.cat((0.7 * torch.ones(self.nllms, 1), 0.3 * torch.ones(self.nllms, 1)), dim=-1)
        elif self.mode == "gt":
            self.output_layer = torch.nn.Linear(self.llm.config.hidden_size, 2)
            self.output_layer.bias.data = self.output_layer.bias.data * 0
        else:
            # self.kproj = torch.nn.Linear(self.llm.config.hidden_size+1+pos_emb_dim, 64)
            self.qproj = torch.nn.Linear(self.llm.config.hidden_size+1+pos_emb_dim, 64)
            self.kproj = torch.nn.Linear(1+pos_emb_dim, 64)
            self.z_hat_proj = torch.nn.Linear(self.outer_dim, 1)
            # self.pos_emb = torch.nn.Embedding(self.nllms, pos_emb_dim)
        self.activation = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.0)
        self.tokenizer = tokenizer
        self.regression = regression
        self.regularisation = reg_factor
        self.epoch = 0
        self.beta_factor = beta_factor
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
        if "roberta" in self.model_path:
            pred_hidden = outputs.last_hidden_state[:, 0]
        else:
            pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        if self.regression == "hardlabel":
            labels = (workers < 0.5).long()
        # pred_hidden = pred_hidden * 0

        if self.mode == "pewcrowd":
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
            latent_dist = self.bottleneck(self.drop(pred_hidden))
            latent_dist = torch.softmax(latent_dist, dim=-1)
            pred_hidden = self.outlayer(latent_dist)
            pred_hidden = torch.clamp(pred_hidden, min=0.0001, max=0.9999)
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
        elif self.mode == "pewcrowdimpxt" or self.mode == "pewcrowdaext":
            latent_dist = self.bottleneck(self.drop(pred_hidden))
            latent_dist = torch.softmax(latent_dist, dim=-1)
            normalised_weight = self.skilllayer(pred_hidden.detach())
            normalised_weight = torch.sigmoid(normalised_weight).view(pred_hidden.size(0), self.nllms, 2)
            if self.epoch < self.freeze_epoch:
                pred_hidden = self.outlayer(latent_dist)
                extra_loss = ((normalised_weight - self.outlayer.weight.data) ** 2).mean()
                normalised_weight = self.outlayer.weight.unsqueeze(0)
            else:
                pred_hidden = (normalised_weight * latent_dist.unsqueeze(1)).sum(dim=-1)
                # extra_loss = ((normalised_weight - self.outlayer.weight.data) ** 2).mean()
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
        elif self.mode == "gt":
            prediction = self.output_layer(pred_hidden)
            loss = torch.nn.functional.cross_entropy(prediction, labels)
        elif "bt" in self.mode:
            label_tokens = inputs['input_ids'][:, 1:].reshape(-1)
            logits = outputs.logits[:, :-1]
            logits = logits.reshape(logits.size(0)*logits.size(1), -1)
            pi_ys = - torch.nn.functional.cross_entropy(logits, label_tokens, reduction="none")
            pi_ys = pi_ys.view(labels.size(0)*2, -1) * inputs["attention_mask"][:, 1:]
            norms = inputs["attention_mask"][:, 1:].sum(dim=-1).view(labels.size(0), 2)
            pi_ys = pi_ys.view(labels.size(0), 2, -1).sum(dim=-1)
            pi_ys = pi_ys / norms
            btvalue = torch.sigmoid(self.beta_factor * (pi_ys[:, 0] - pi_ys[:, 1]))
            if self.mode == "btgt":
                workers = workers.mean(dim=-1)
                if self.regression == "skill":
                    loss = - workers * torch.log(btvalue) - (1 - workers) * torch.log(1 - btvalue)
                else:
                    labels = (workers > 0.5).long()
                    btvalue = torch.clamp(btvalue, min=0.0001, max=0.9999)
                    loss = - labels * torch.log(btvalue) - (1 - labels) * torch.log(1 - btvalue)
                loss = loss.mean()
            elif self.mode == "btcrowd":
                btvalue = torch.cat([btvalue.unsqueeze(-1), 1-btvalue.unsqueeze(-1)], dim=-1)
                pred_hidden = self.outlayer(btvalue).view(workers.size(0)*self.nllms, 2)
                if self.regression == "skill":
                    pred_hidden = torch.softmax(pred_hidden, dim=-1)
                    workers = workers.view(-1)
                    loss = - workers * torch.log(pred_hidden[:, 0]) - (1 - workers) * torch.log(1 - pred_hidden[:, 0])
                else:
                    loss = torch.nn.functional.cross_entropy(pred_hidden, labels.view(-1))
                loss = loss.mean()
            elif self.mode == "btskillagg":
                btvalue = torch.cat([btvalue.unsqueeze(-1), 1-btvalue.unsqueeze(-1)], dim=-1)
                pred_hidden = self.outlayer(btvalue)
                pred_hidden = torch.clamp(pred_hidden, min=0.0001, max=0.9999)
                if self.regression == "skill":
                    loss = - workers * torch.log(pred_hidden) - (1 - workers) * torch.log(1 - pred_hidden)
                else:
                    loss = - labels * torch.log(pred_hidden) - (1 - labels) * torch.log(1 - pred_hidden)
                loss = loss.mean()
                loss += self.regularisation * ((self.outlayer.weight[:,0] - self.outlayer.weight[:,1]) ** 2).mean()
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
        if "roberta" in self.model_path:
            pred_hidden = outputs.last_hidden_state[:, 0]
        else:
            pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]

        # pred_hidden = pred_hidden * 0
        if self.mode == "pewcrowd":
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            pred_hidden = 1 - torch.softmax(self.outlayer(prediction).view(prediction.size(0)*self.nllms, 2), dim=-1)
        elif self.mode == "pewcrowdimp" or self.mode == "pewcrowdae":
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            pred_hidden = self.outlayer(prediction).view(prediction.size(0)*self.nllms, 1)
            normalised_weight = self.outlayer.weight.data
            pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)

            # EM E-step
            if withEM:
                sigma = prediction[:, 0]
                p_r_0 = normalised_weight[:, 0]
                p_r_1 = 1 - normalised_weight[:, 1]
                numerator = torch.log(p_r_0).unsqueeze(0) * (1 - labels) + torch.log(1 - p_r_0) * labels
                numerator = sigma * torch.exp(numerator.sum(dim=-1))
                denominator = torch.log(p_r_1).unsqueeze(0) * labels + torch.log(1 - p_r_1) * (1 - labels)
                denominator = (1 - sigma) * torch.exp(denominator.sum(dim=-1))
                # prediction = (numerator < denominator).float().unsqueeze(-1)
                prediction = (denominator / (numerator + denominator)).float().unsqueeze(-1)
                prediction = torch.cat([1-prediction, prediction], dim=-1)
        elif self.mode == "pewcrowdimpxt" or self.mode == "pewcrowdaext":
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
                prediction = (denominator / (numerator + denominator)).float().unsqueeze(-1)
                # prediction = (numerator < denominator).float().unsqueeze(-1)
                prediction = torch.cat([1-prediction, prediction], dim=-1)
        elif self.mode == "pewcrowdaepost":
            prediction = self.bottleneck(pred_hidden)
            prediction = torch.softmax(prediction, dim=-1)
            # pred_hidden = self.outlayer(prediction).view(prediction.size(0)*self.nllms, 1)
            normalised_weight = torch.softmax(self.outlayer.weight, -1).unsqueeze(0)
            pred_hidden = (prediction.unsqueeze(1) * normalised_weight).sum(dim=-1)
            normalised_weight_corr = torch.softmax(self.correlationlayer.weight, -1).unsqueeze(0)
            pred_hidden = (pred_hidden.unsqueeze(1) * normalised_weight_corr).sum(dim=-1)
            pred_hidden = pred_hidden.view(prediction.size(0)*self.nllms, 1)
            pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)

        elif self.mode == "gt":
            # pred_hidden = torch.cat([pred_hidden, workers], dim=-1)
            prediction = torch.softmax(self.output_layer(pred_hidden), dim=-1)
        elif "bt" in self.mode:
            label_tokens = inputs['input_ids'][:, 1:].reshape(-1)
            logits = outputs.logits[:, :-1]
            logits = logits.reshape(logits.size(0)*logits.size(1), -1)
            pi_ys = - torch.nn.functional.cross_entropy(logits, label_tokens, reduction="none")
            pi_ys = pi_ys.view(labels.size(0)*2, -1) * inputs["attention_mask"][:, 1:]
            norms = inputs["attention_mask"][:, 1:].sum(dim=-1).view(labels.size(0), 2)
            pi_ys = pi_ys.view(labels.size(0), 2, -1).sum(dim=-1)
            pi_ys = pi_ys / norms
            prediction = torch.sigmoid(self.beta_factor * (pi_ys[:, 0] - pi_ys[:, 1]))
            pred_hidden = torch.cat([prediction.unsqueeze(-1), 1-prediction.unsqueeze(-1)], dim=-1)
            # EM E-step
            if withEM and "skillagg" in self.mode:
                pred_hidden = self.outlayer(pred_hidden).view(prediction.size(0)*self.nllms, 1)
                pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)
                normalised_weight = self.outlayer.weight.data
                p_r_0 = normalised_weight[:, 0]
                p_r_1 = 1 - normalised_weight[:, 1]
                numerator = torch.log(p_r_0).unsqueeze(0) * (1 - labels) + torch.log(1 - p_r_0) * labels
                numerator = prediction * torch.exp(numerator.sum(dim=-1))
                denominator = torch.log(p_r_1).unsqueeze(0) * labels + torch.log(1 - p_r_1) * (1 - labels)
                denominator = (1 - prediction) * torch.exp(denominator.sum(dim=-1))
                prediction = (numerator / (numerator + denominator)).float().unsqueeze(-1)
                prediction = torch.cat([prediction, 1-prediction], dim=-1)
            else:
                prediction = pred_hidden
                if "skillagg" in self.mode:
                    pred_hidden = self.outlayer(pred_hidden).view(prediction.size(0)*self.nllms, 1)
                    pred_hidden = torch.cat([1-pred_hidden, pred_hidden], dim=-1)
                else:
                    pred_hidden = 1 - torch.softmax(self.outlayer(prediction).view(prediction.size(0)*self.nllms, 2), dim=-1)
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
