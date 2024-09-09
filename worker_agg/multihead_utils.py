import os
import math
import time
from collections import OrderedDict
from typing import Dict
import copy

import numpy as np
from tqdm import tqdm
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
                 head_idx: int,
                 unfreeze_layers, 
                 num_workers: int,
                 hidden_size_ow: int, 
                 gpt_hidden_size: int, 
                 device,
                 trans_ln_f,
                 dropout_prob: float = 0.1):
        super(SSLHead, self).__init__()
        self.head_idx = head_idx
        # self.lm_unfrozen = copy.deepcopy(unfreeze_layers).to(device)
        setattr(self, f"unfrozen_{head_idx}", copy.deepcopy(unfreeze_layers).to(device))
        # self.ow_l1 = nn.Linear(num_workers - 1, hidden_size_ow).to(device)
        setattr(self, f"ow_l1_{head_idx}", nn.Linear(num_workers - 1, hidden_size_ow).to(device))
        # self.ow_l2 = nn.Linear(gpt_hidden_size + hidden_size_ow, 1).to(device)
        setattr(self, f"ow_l2_{head_idx}", nn.Linear(gpt_hidden_size + hidden_size_ow, 1).to(device))
        # self.trans_ln_f = trans_ln_f
        setattr(self, f"trans_ln_f_{head_idx}", copy.deepcopy(trans_ln_f))
        self.dropout_prob = dropout_prob
    
    def forward(self, pred_hidden: torch.Tensor, 
                ow_ests: torch.Tensor, 
                attention_mask: torch.Tensor, 
                insizes: torch.Tensor) -> torch.Tensor:
        for layer in getattr(self, f"unfrozen_{self.head_idx}"):
            pred_hidden = layer(pred_hidden, attention_mask=attention_mask)[0]
        # apply layer norm -- see forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
        pred_hidden = getattr(self, f"trans_ln_f_{self.head_idx}")(pred_hidden)

        # Extract the hidden states at the positions specified by 'insizes'
        pred_hidden = pred_hidden[torch.arange(insizes.size(0)), insizes]
        # copy_outputs.append(pred_hidden)

        ow_op = getattr(self, f"ow_l1_{self.head_idx}")(ow_ests)
        ow_op = torch.relu(ow_op)
        # apply dropout
        ow_op = nn.functional.dropout(ow_op, p=self.dropout_prob, training=self.training)
        pred_hidden = torch.cat((pred_hidden, ow_op), dim=1)
        pred_hidden = getattr(self, f"ow_l2_{self.head_idx}")(pred_hidden)
        return pred_hidden

class StdHead(nn.Module):
    def __init__(self, 
                 head_idx: int,
                 unfreeze_layers, 
                 gpt_hidden_size: int, 
                 device,
                 trans_ln_f,):
        super(StdHead, self).__init__()
        # self.lm_unfrozen = copy.deepcopy(unfreeze_layers).to(device)
        # self.op_layer = nn.Linear(gpt_hidden_size, 1).to(device)
        # self.trans_ln_f = trans_ln_f
        self.head_idx = head_idx
        setattr(self, f"unfrozen_{head_idx}", copy.deepcopy(unfreeze_layers).to(device))
        setattr(self, f"op_layer_{head_idx}", nn.Linear(gpt_hidden_size, 1).to(device))
        setattr(self, f"trans_ln_f_{head_idx}", copy.deepcopy(trans_ln_f))
    
    def forward(self, pred_hidden: torch.Tensor, 
                attention_mask: torch.Tensor, 
                insizes: torch.Tensor) -> torch.Tensor:
        for layer in getattr(self, f"unfrozen_{self.head_idx}"):
            pred_hidden = layer(pred_hidden, attention_mask=attention_mask)[0]
        # apply layer norm -- see forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
        pred_hidden = getattr(self, f"trans_ln_f_{self.head_idx}")(pred_hidden)

        # Extract the hidden states at the positions specified by 'insizes'
        pred_hidden = pred_hidden[torch.arange(insizes.size(0)), insizes]
        # copy_outputs.append(pred_hidden)

        pred_hidden = getattr(self, f"op_layer_{self.head_idx}")(pred_hidden)
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
        loss_fn_type: str = 'ce',
        do_ssl: bool = True,
        hidden_size_ow: int = 100,
    ):
        super(MultiHeadNet, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.num_workers = num_workers
        self.dropout_prob = dropout_prob
        assert loss_fn_type in ['mse','ce'], "Loss function type must be either 'mse' or 'ce'"
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
                    head_idx=i,
                    unfreeze_layers=llm.transformer.h[-n_unfreeze:],
                    num_workers=num_workers,
                    hidden_size_ow=hidden_size_ow,
                    gpt_hidden_size=llm.config.hidden_size,
                    device=self.device,
                    trans_ln_f=llm.transformer.ln_f,
                ) for i in range(num_workers)
            ])
        else:
            self.heads = nn.ModuleList([
                StdHead(
                    head_idx=i,
                    unfreeze_layers=llm.transformer.h[-n_unfreeze:],
                    gpt_hidden_size=llm.config.hidden_size,
                    device=self.device,
                    trans_ln_f=llm.transformer.ln_f,
                ) for i in range(num_workers)
            ])

    def forward(self, 
                inputs: tuple,
                predict_gt: bool=False,
                testing: bool=False) -> torch.Tensor:
        if self.do_ssl:
            inputs, ests = inputs
        assert type(inputs) == dict, "Inputs must be a dictionary"
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
        if self.loss_fn_type == 'ce':
            # need preds to be of shape (batch_size, 2, num_workers)
            # concatenate preds with -preds
            if not predict_gt:
                preds = preds.unsqueeze(1)
                preds = torch.cat((-preds, preds), dim=1)
            else:
                # convert to probabilities
                ssl_preds = torch.sigmoid(preds)
                # avg over workers
                preds = torch.mean(ssl_preds, dim=1)
        elif self.loss_fn_type == 'mse' and predict_gt:
            # preds are probabilities already
            ssl_preds = preds.clone()
            preds = torch.mean(preds, dim=1)
        if testing:
            return preds, ssl_preds
        return preds
    
    def set_requires_grad_for_heads(self, active_heads):
        print('set requires_grad for heads')
        print(f"active heads: {active_heads}")
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
                log_interval: int=100, patience: int=2, loss_fn_type='ce',
                probs: bool=False
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
        self.probs = probs
        # if self.loss_fn_type == 'bce':
        #     self.criterion = nn.BCEWithLogitsLoss()
        if self.loss_fn_type == 'mse':
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
        ## Optimiser
        no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters =[]
        # for head in self.model.heads:
        #     optimizer_grouped_parameters += [
        #         {
        #             "params": [p for n, p in head.named_parameters() if not any(nd in n for nd in no_decay)],
        #             "weight_decay": self.weight_decay,
        #         },
        #         {
        #             "params": [p for n, p in head.named_parameters() if any(nd in n for nd in no_decay)],
        #             "weight_decay": 0.0,
        #         },
        #     ]
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
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        num_warmup_steps = self.num_warmup_steps * max_train_steps
        self.lr_scheduler = get_scheduler(
                name=self.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_train_steps,
            )

    def run(self):
        self.init_opts_schedulers()

        # Train loop
        best_val_losses = [float('inf') for _ in range(self.num_workers)]
        best_epochs = [0 for _ in range(self.num_workers)]
        epochs_no_improve = [0 for _ in range(self.num_workers)]
        # self.eval_one_epoch()
        for epoch in range(self.num_train_epochs):
            self.model.train()
            self.train_one_epoch(epoch,)
            # if args.split < 1.0:
            # self.model.eval()
            val_losses = self.eval_one_epoch()

            # Early stopping
            for i in range(self.num_workers):
                if i not in self.active_heads:
                    assert np.isclose(val_losses[i], best_val_losses[i], rtol=1e-3)
                    continue
                if val_losses[i] < best_val_losses[i]:
                    best_val_losses[i] = val_losses[i]
                    best_epochs[i] = epoch
                    epochs_no_improve[i] = 0
                else:
                    epochs_no_improve[i] += 1

                self.save_checkpoint(epoch, i)

            to_remove = []
            for i in self.active_heads:
                if epochs_no_improve[i] >= self.patience:
                    self.logging(f'For model {i}, early stopping and loading {best_epochs[i]}',
                                 self.model_dir + '/train.log')
                    self.load_checkpoint(best_epochs[i], head_idx=i)
                    to_remove.append(i)
            for i in to_remove:
                self.active_heads.remove(i)
            # self.active_heads = [0,4]
            # self.model.set_requires_grad_for_heads(self.active_heads)
            self.set_require_grad()
            if len(self.active_heads) == 0:
                print("All heads have stopped improving. Training complete.")
                print("Best epochs: ", best_epochs)
                self.save_checkpoint(epoch)
                break

            # if epochs_no_improve >= self.patience:
            #     self.logging(f'Early stopping at epoch {epoch}', 
            #                 self.model_dir + '/train.log')
            #     self.load_checkpoint(best_epoch)
            #     break
        for i in self.active_heads:
            self.load_checkpoint(best_epochs[i], head_idx=i)    

    def set_require_grad(self,):
        """Set requires_grad=True only for the selected heads."""
        # print(f"len(self.model.heads): {len(self.model.heads)}")
        # print(f"self.active_heads: {self.active_heads}")
        for i, head in enumerate(self.model.heads):
            # print(f"head {i}:")
            if i in self.active_heads:
                for name, param in head.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in head.named_parameters():
                    param.requires_grad = False
    
    def check_require_grad(self,):
        """Set requires_grad=True only for the selected heads."""
        # print(f"len(self.model.heads): {len(self.model.heads)}")
        # print(f"self.active_heads: {self.active_heads}")
        for i, head in enumerate(self.model.heads):
            # print(f"head {i}:")
            if i in self.active_heads:
                for name, param in head.named_parameters():
                    try:
                        assert param.requires_grad == True
                    except:
                        print(f"param {name} requires_grad: {param.requires_grad}")
                        raise AssertionError
            else:
                for name, param in head.named_parameters():
                    try:
                        assert param.requires_grad == False
                    except:
                        print(f"param {name} requires_grad: {param.requires_grad}")
                        raise AssertionError

    def train_one_epoch(self, epoch,):
        self.optimizer.zero_grad()
        trainsize = len(self.train_dataloader)
        start = time.time()
        for i, batch in enumerate(self.train_dataloader):
            inputs, labels = batch
            outputs = self.model(inputs)
            assert len(self.active_heads) > 0, "No active heads"
            # loss = self.criterion(outputs[:,self.active_heads], labels[:,self.active_heads].float())
            if self.loss_fn_type == 'ce':
                loss = self.criterion(outputs, labels)
            elif self.loss_fn_type == 'mse':
                loss = self.criterion(outputs, labels.float())
            loss = loss / self.gradient_accumulation_steps
            # self.check_require_grad()
            loss.backward()
            # train_losses = []
            # total_loss = 0
            # for j in self.active_heads:
            #     loss = self.criterion(outputs[:,j], labels[:,j].float())
            #     loss = loss / self.gradient_accumulation_steps
            #     total_loss += loss
            #     train_losses.append(loss.item()*self.gradient_accumulation_steps)
            
            # total_loss.backward()
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if (i + 1) % self.log_interval == 0:
                elapsed_time = time.time() - start
                logfile = self.model_dir + '/train.log'
                self.logging(f"Epoch {epoch} | Batch {i+1}/{trainsize} | time {elapsed_time}", 
                        logfile)
                # formatted_losses = ', '.join([f'{loss:.4f}' for loss in train_losses])
                self.logging(f"active heads: {self.active_heads} | loss: {loss:.4f}", 
                        logfile)

    def eval_one_epoch(self):
        self.check_require_grad()
        self.model.eval() # Set model to evaluate mode
        hits = [0 for _ in range(self.num_workers)]
        total = [0 for _ in range(self.num_workers)]
        total_loss = [0 for _ in range(self.num_workers)]
        start = time.time()
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            inputs, labels = batch
            # forward pass
            preds = self.model(inputs)
            # print("shape of preds: ", preds.shape)
            # print("shape of labels: ", labels.shape)
            # calculate loss for all workers irrespective of active heads
            for j in range(self.num_workers):
                if self.loss_fn_type == 'ce':
                    if not self.probs:
                        loss = self.criterion(preds[:,:,j], labels[:,j])
                    else:
                        loss = self.criterion(preds[:,:,j], labels[:,:,j])
                elif self.loss_fn_type == 'mse':
                    loss = self.criterion(preds[:,j], labels[:,j].float())
                total_loss[j] += loss.item() * labels.size(0)
                if self.loss_fn_type == 'ce':
                    if self.probs:
                        hits[j] += sum((labels[:,1,j]>0.5).long().view(-1) 
                                    == (preds[:,1,j] > 0).long().view(-1)).item()
                    else:
                        hits[j] += sum(labels[:,j].view(-1) == (preds[:,1,j] > 0).long().view(-1)).item()
                elif self.loss_fn_type == 'mse':
                    hits[j] += sum((labels[:,j]>0.5).long().view(-1) 
                                   == (preds[:,j] > 0.5).long().view(-1)).item()
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
        # head.load_state_dict(torch.load(f'{checkpoint_dir}/pytorch_model.pt'), strict=False)
        head.load_state_dict(torch.load(f'{checkpoint_dir}/pytorch_model.pt'))