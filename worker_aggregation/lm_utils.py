import os
import math
import time
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class LMplusOneLayer(nn.Module):
    def __init__(
        self,
        model_path: str,
        seed: int,
        dropout_prob: float = 0.1,
        cache_dir: str = "scratch/cache"
    ):
        super(LMplusOneLayer, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.dropout_prob = dropout_prob

        # Load the pre-trained language model
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir
        )

        # Initialize additional layers
        self.output_layer = nn.Linear(self.llm.config.hidden_size, 1)
        self.activation = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _initialize_weights(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        nn.init.kaiming_normal_(self.output_layer.weight, 
                                nonlinearity='relu', generator=generator)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, 
                inputs: Dict[str, torch.Tensor], ):
        attention_mask = inputs["attention_mask"]
        outputs = self.llm(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]

        # Apply deterministic dropout
        if self.training:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            dropout_mask = (torch.rand(pred_hidden.shape, generator=generator) > self.dropout_prob).float().to(pred_hidden.device)
            pred_hidden = pred_hidden * dropout_mask / (1 - self.dropout_prob)

        logits = self.output_layer(pred_hidden)
        return logits

class FinetuneLM:
    def __init__(self, model, train_dataloader, val_dataloader,
                model_dir: str,
                lr: float=0.001, weight_decay: float=1e-5, 
                gradient_accumulation_steps: int=1, num_warmup_steps: float=0.03,
                num_train_epochs: int=10, lr_scheduler_type: str='cosine',
                log_interval: int=100) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_dir = model_dir
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.num_train_epochs = num_train_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.log_interval = log_interval
    
    def logging(self, string: str, 
                logfilename: str, 
                print_str=True, write_str=True):
        if print_str:
            print(string)
        if write_str:
            with open(logfilename, 'a+') as f_log:
                f_log.write(string + '\n')

    def run(self):
        ## Optimiser
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        num_warmup_steps = num_warmup_steps * max_train_steps

        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Train loop
        for epoch in range(self.num_train_epochs):
            self.model.train()
            self.train_one_epoch(epoch,)
            # if args.split < 1.0:
            self.model.eval()
            self.eval_one_epoch()

            # current_lr = optimizer.param_groups[0]["lr"]
            self.save_checkpoint(epoch)

    def train_one_epoch(self, epoch,):
        self.optimizer.zero_grad()
        trainsize = len(self.train_dataloader)
        start = time.time()
        for i, batch in enumerate(self.train_dataloader):
            inputs, labels = batch
            loss = self.model(
                inputs,
                labels,
            )
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if (i + 1) % self.log_interval == 0:
                elasped_time = time.time() - start
                loss = loss.item() * self.gradient_accumulation_steps
                logfile = self.model_dir + '/train.log'
                self.logging(f"Epoch {epoch} | Batch {i+1}/{trainsize} | loss: {loss} | time {elasped_time}", 
                        logfile)

    def eval_one_epoch(self):
        hits = 0
        total = 0
        for i, batch in enumerate(self.val_dataloader):
            inputs, labels = batch
            preds = self.model.predict(
                inputs,
                labels,
            )
            hits += sum(labels.view(-1) == preds.view(-1))
            total += preds.size(0)
        print("Accuracy: {:.2f}".format(hits/total))

    def save_checkpoint(self, epoch):
        fulloutput = os.path.join(self.model_dir, "checkpoint.{}".format(epoch))
        os.system(f"mkdir -p {fulloutput}")
        checkpoint = OrderedDict()
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v
        torch.save(checkpoint, f'{fulloutput}/pytorch_model.pt')
        # save tokenizer
        self.model.tokenizer.save_pretrained(fulloutput)
        return checkpoint