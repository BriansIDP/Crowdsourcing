import os
import math
import time
from collections import OrderedDict
from typing import Dict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from transformers import AutoModelForCausalLM

class MultiHeadBase(nn.Module):
    def __init__(self, frozen_part, trans_wte,
                 trans_wpe, trans_drop, device):
        super(MultiHeadBase, self).__init__()
        self.frozen_part = frozen_part
        self.trans_wte = trans_wte
        self.trans_wpe = trans_wpe
        self.trans_drop = trans_drop
        self.device = device

    def forward(self, inputs):
        # Ensure inputs are on the correct device
        attention_mask = inputs["attention_mask"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        # Get token embeddings (wte) and positional embeddings (wpe)
        token_embeddings = self.trans_wte(input_ids)  # Word/Token Embeddings
        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=self.device).unsqueeze(0)
        position_embeddings = self.trans_wpe(position_ids)  # Positional Embeddings
        # Combine token and positional embeddings
        hidden_states = token_embeddings + position_embeddings
        # ChatGPT says applying dropout is standard practice even though this is from the frozen part
        hidden_states = self.trans_drop(hidden_states)

        # Prepare attention mask for transformer blocks
        # See forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
        # specifically, lines 815 and 823
        # TODO: this might not work for all models
        insizes = attention_mask.clone().sum(dim=-1) - 1
        attention_mask = attention_mask[:, None, None, :].to(device=self.device).float()
        attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        for layer in self.frozen_part:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states, attention_mask, insizes

class SSLHead(nn.Module):
    def __init__(self, 
                 unfreeze_layers, 
                 hidden_size_ow: int, 
                 gpt_hidden_size: int, 
                 device,
                 trans_ln_f,
                 dropout_prob: float = 0.1):
        super(SSLHead, self).__init__()
        self.lm_unfrozen = copy.deepcopy(unfreeze_layers).to(device)
        self.ow_l1 = nn.Linear(gpt_hidden_size - 1, hidden_size_ow).to(device)
        self.ow_l2 = nn.Linear(gpt_hidden_size + hidden_size_ow, 1).to(device)
        self.trans_ln_f = trans_ln_f
        self.dropout_prob = dropout_prob
    
    def forward(self, pred_hidden: torch.Tensor, 
                ow_ests: torch.Tensor, 
                attention_mask: torch.Tensor, 
                insizes: torch.Tensor) -> torch.Tensor:
        for layer in self.lm_unfrozen:
            pred_hidden = layer(pred_hidden, attention_mask=attention_mask)[0]
        # apply layer norm -- see forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
        pred_hidden = self.trans_ln_f(pred_hidden)

        # Extract the hidden states at the positions specified by 'insizes'
        pred_hidden = pred_hidden[torch.arange(insizes.size(0)), insizes]
        # copy_outputs.append(pred_hidden)

        ow_op = self.ow_l1(ow_ests)
        ow_op = torch.relu(ow_op)
        # apply dropout
        ow_op = nn.functional.dropout(ow_op, p=self.dropout_prob, training=self.training)
        pred_hidden = torch.cat((pred_hidden, ow_op), dim=1)
        pred_hidden = self.ow_l2(pred_hidden)
        return pred_hidden

class StdHead(nn.Module):
    def __init__(self, 
                 unfreeze_layers, 
                 gpt_hidden_size: int, 
                 device,
                 trans_ln_f,):
        super(SSLHead, self).__init__()
        self.lm_unfrozen = copy.deepcopy(unfreeze_layers).to(device)
        self.op_layer = nn.Linear(gpt_hidden_size, 1).to(device)
        self.trans_ln_f = trans_ln_f
    
    def forward(self, pred_hidden: torch.Tensor, 
                attention_mask: torch.Tensor, 
                insizes: torch.Tensor) -> torch.Tensor:
        for layer in self.lm_unfrozen:
            pred_hidden = layer(pred_hidden, attention_mask=attention_mask)[0]
        # apply layer norm -- see forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
        pred_hidden = self.trans_ln_f(pred_hidden)

        # Extract the hidden states at the positions specified by 'insizes'
        pred_hidden = pred_hidden[torch.arange(insizes.size(0)), insizes]
        # copy_outputs.append(pred_hidden)

        pred_hidden = self.op_layer(pred_hidden)
        return pred_hidden

class MultiHeadNet(nn.Module):
    def __init__(
        self,
        num_workers: int,
        model_path: str,
        seed: int,
        dropout_prob: float = 0.1,
        cache_dir: str = "scratch/cache",
        n_unfreeze: int = 0,
        loss_fn_type: str = 'bce',
        do_ssl: bool = True,
        hidden_size_ow: int = 100,
    ):
        super(MultiHeadNet, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.num_workers = num_workers
        self.dropout_prob = dropout_prob
        assert loss_fn_type in ['mse','bce'], "Loss function type must be either 'mse' or 'bce'"
        self.loss_fn_type = loss_fn_type

        # Determine device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained language model and move to the correct device
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir
        ).to(self.device)
        self.n_unfreeze = n_unfreeze
        for param in llm.parameters():
            param.requires_grad = False
        
        self.base = MultiHeadBase(frozen_part=llm.transformer.h[:-n_unfreeze],
                                  trans_wte=llm.transformer.wte,
                                  trans_wpe=llm.transformer.wpe,
                                  trans_drop=llm.transformer.drop,
                                  device=self.device)

        self.do_ssl = do_ssl
        if do_ssl:
            self.hidden_size_ow = hidden_size_ow
            self.heads = nn.ModuleList([
                SSLHead(
                    unfreeze_layers=llm.transformer.h[-n_unfreeze:],
                    hidden_size_ow=hidden_size_ow,
                    gpt_hidden_size=llm.config.hidden_size,
                    device=self.device,
                    trans_ln_f=llm.transformer.ln_f,
                ) for _ in range(num_workers)
            ])
        else:
            self.heads = nn.ModuleList([
                StdHead(
                    unfreeze_layers=llm.transformer.h[-n_unfreeze:],
                    gpt_hidden_size=llm.config.hidden_size,
                    device=self.device,
                    trans_ln_f=llm.transformer.ln_f,
                ) for _ in range(num_workers)
            ])

    def forward(self, 
                inputs: tuple,
                predict_gt: bool=False) -> torch.Tensor:
        inputs, ests = inputs
        # passing input through the base
        hidden_states, attention_mask, insizes = self.base(inputs)

        # pass through each head
        preds = []
        for i in range(self.num_workers):
            pred_hidden = hidden_states.clone()
            if self.do_ssl:
                other_ids = torch.arange(self.num_workers)[torch.arange(self.num_workers) != i]
                ow_ests = ests[:,other_ids].clone().float()
                pred_hidden = self.heads[i](pred_hidden, ow_ests, attention_mask, insizes)
            else:
                pred_hidden = self.heads[i](pred_hidden, attention_mask, insizes)
            preds.append(pred_hidden)
        preds = torch.cat(preds, dim=1)
        if predict_gt:
            if self.loss_fn_type == 'bce':
                preds = torch.sigmoid(preds)
            preds = torch.mean(preds, dim=1)
        return preds
    
    def set_requires_grad_for_heads(self, active_heads):
        """Set requires_grad=True only for the selected heads."""
        for i, head in enumerate(self.heads):
            if i in active_heads:
                for param in head.parameters():
                    param.requires_grad = True
            else:
                for param in head.parameters():
                    param.requires_grad = False

class FinetuneMultiHeadNet():
    def __init__(self, model, train_dataloader, val_dataloader,
                model_dir: str, num_workers: int,
                lr: float=0.001, weight_decay: float=1e-5, 
                gradient_accumulation_steps: int=1, num_warmup_steps: float=0.03,
                num_train_epochs: int=10, lr_scheduler_type: str='cosine',
                log_interval: int=100, patience: int=2, loss_fn_type='bce',
                ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_dir = model_dir
        self.num_workers = num_workers
        self.active_heads = list(range(num_workers))
        self.model.set_requires_grad_for_heads(self.active_heads)
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.num_train_epochs = num_train_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.log_interval = log_interval
        self.patience = patience
        self.loss_fn_type = loss_fn_type
        if self.loss_fn_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_fn_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_fn_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Loss function {self.loss_fn_type} not recognised')
    
    def logging(self, string: str, 
                logfilename: str, 
                print_str=True, write_str=True):
        if print_str:
            print(string)
        if write_str:
            with open(logfilename, 'a+') as f_log:
                f_log.write(string + '\n')
    
    def init_opts_schedulers(self):
        ## Optimisers
        no_decay = ["bias", "LayerNorm.weight"]
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        num_warmup_steps = self.num_warmup_steps * max_train_steps

        self.optimizers = []
        self.lr_schedulers = []
        for i in range(self.num_workers):
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.heads[i].named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.heads[i].named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizers.append(optim.AdamW(optimizer_grouped_parameters, lr=self.lr))

        self.lr_schedulers.append(
            get_scheduler(
                name=self.lr_scheduler_type,
                optimizer=self.optimizers[i],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_train_steps,
            )
        )

    def run(self):
        self.init_opts_schedulers()

        # Train loop
        best_val_losses = [float('inf') for _ in range(self.num_workers)]
        best_epochs = [0 for _ in range(self.num_workers)]
        epochs_no_improve = [0 for _ in range(self.num_workers)]
        for epoch in range(self.num_train_epochs):
            self.model.train()
            self.train_one_epoch(epoch,)
            # if args.split < 1.0:
            self.model.eval()
            val_losses = self.eval_one_epoch()

            # Early stopping
            for i in range(self.num_workers):
                if i not in self.active_heads:
                    assert torch.isclose(val_losses[i], best_val_losses[i], rtol=1e-3)
                    continue
                if val_losses[i] < best_val_losses[i]:
                    best_val_losses[i] = val_losses[i]
                    best_epochs[i] = epoch
                    epochs_no_improve[i] = 0
                else:
                    epochs_no_improve[i] += 1

            self.save_checkpoint(epoch)

            to_remove = []
            for i in self.active_heads:
                if epochs_no_improve[i] >= self.patience:
                    self.logging(f'For model {i}, early stopping and loading {best_epochs[i]}',
                                 self.model_dir + '/train.log')
                    self.load_checkpoint(best_epochs[i], head_idx=i)
                    to_remove.append(i)
            for i in to_remove:
                self.active_heads.remove(i)
            self.model.set_requires_grad_for_heads(self.active_heads)
            if len(self.active_heads) == 0:
                print("All heads have stopped improving. Training complete.")
                print("Best epochs: ", best_epochs)
                break

            # if epochs_no_improve >= self.patience:
            #     self.logging(f'Early stopping at epoch {epoch}', 
            #                 self.model_dir + '/train.log')
            #     self.load_checkpoint(best_epoch)
            #     break

    def train_one_epoch(self, epoch,):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        trainsize = len(self.train_dataloader)
        start = time.time()
        for i, batch in enumerate(self.train_dataloader):
            inputs, labels = batch
            outputs = self.model(inputs)
            assert len(self.active_heads) > 0, "No active heads"
            train_losses = []
            for j in self.active_heads:
                loss = self.criterion(outputs[:,j], labels[:,j].float())
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                train_losses.append(loss.item()*self.gradient_accumulation_steps)

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.optimizers[j].step()
                    self.lr_schedulers[j].step()
                    self.optimizers[j].zero_grad()
            if (i + 1) % self.log_interval == 0:
                elapsed_time = time.time() - start
                logfile = self.model_dir + '/train.log'
                self.logging(f"Epoch {epoch} | Batch {i+1}/{trainsize} | time {elapsed_time}", 
                        logfile)
                self.logging(f"active heads: {self.active_heads} | losses: {train_losses:.4f}",)

    def eval_one_epoch(self):
        self.model.eval() # Set model to evaluate mode
        hits = [0 for _ in range(self.num_workers)]
        total = [0 for _ in range(self.num_workers)]
        total_loss = [0 for _ in range(self.num_workers)]
        start = time.time()
        for i, batch in enumerate(self.val_dataloader):
            inputs, labels = batch
            # forward pass
            preds = self.model(inputs)
            # print("shape of preds: ", preds.shape)
            # print("shape of labels: ", labels.shape)
            # calculate loss for all workers irrespective of active heads
            for j in range(self.num_workers):
                loss = self.criterion(preds[:,j], labels[:,j].float())
                total_loss[j] += loss.item() * labels.size(0)
                if self.loss_fn_type == 'bce':
                    hits[j] += sum(labels[:,j].view(-1) == (preds[:,j] > 0).long().view(-1))
                elif self.loss_fn_type == 'mse':
                    hits[j] += sum(labels[:,j].view(-1) == (preds[:,j] > 0.5).long().view(-1))
                total[j] += preds.size(0)
        elapsed_time = time.time() - start
        avg_losses = [total_loss[j] / total[j] for j in range(self.num_workers)]
        accs = [hits[j] / total[j] for j in range(self.num_workers)]
        self.logging(f"Val acc local: {accs}", 
                    self.model_dir + '/train.log')
        self.logging(f"Val loss local: {avg_losses}",
                    self.model_dir + '/train.log')
        self.logging(f"Time taken: {elapsed_time:.3f}",
                    self.model_dir + '/train.log')
        return avg_losses

    def save_checkpoint(self, epoch, head_idx=None):
        # Determine the directory and filename for the checkpoint
        if head_idx is not None:
            checkpoint_dir = os.path.join(self.model_dir, f"checkpoint_head{head_idx}_epoch{epoch}")
            head = self.model.heads[head_idx]  # Get the specific head
        else:
            checkpoint_dir = os.path.join(self.model_dir, f"checkpoint.{epoch}")
            head = self.model  # Fall back to the entire model if head_idx is None
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # checkpoint = OrderedDict()
        # for k, v in head.named_parameters():  # Iterate over the specific head's parameters
        #     if v.requires_grad:
        #         checkpoint[k] = v
        
        torch.save(head.state_dict(), f'{checkpoint_dir}/pytorch_model.pt')
        
        # return checkpoint

    def load_checkpoint(self, best_epoch, head_idx=None):
        # Determine the directory and filename for the checkpoint
        if head_idx is not None:
            checkpoint_dir = os.path.join(self.model_dir, f"checkpoint_head{head_idx}_epoch{best_epoch}")
            head = self.model.heads[head_idx]  # Get the specific head
        else:
            checkpoint_dir = os.path.join(self.model_dir, f"checkpoint.{best_epoch}")
            head = self.model  # Fall back to the entire model if head_idx is None
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Model checkpoint {checkpoint_dir} does not exist")
        
        # Load the state dict
        head.load_state_dict(torch.load(f'{checkpoint_dir}/pytorch_model.pt'), strict=False)