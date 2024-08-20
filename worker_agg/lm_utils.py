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
from transformers import AutoTokenizer

class LMplusOneLayer(nn.Module):
    def __init__(
        self,
        model_path: str,
        seed: int,
        dropout_prob: float = 0.1,
        cache_dir: str = "scratch/cache",
        no_freeze_lm: bool = True,
        n_unfreeze: int = 0,
    ):
        super(LMplusOneLayer, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.dropout_prob = dropout_prob

        # Determine device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained language model and move to the correct device
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir
        ).to(self.device)
        self.no_freeze_lm = no_freeze_lm
        self.n_unfreeze = n_unfreeze
        if not no_freeze_lm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            # Unfreeze the last n layers of GPT-2
            for layer in self.llm.transformer.h[-n_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # no freezing
            pass

        # Initialize additional layers and move them to the correct device
        self.output_layer = nn.Linear(self.llm.config.hidden_size, 1).to(self.device)
        # self.activation = nn.ReLU().to(self.device)

        # Initialize weights
        self._initialize_weights()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _initialize_weights(self):
        generator = torch.Generator()  # Ensure generator is on the correct device
        generator.manual_seed(self.seed)
        # Manually generate random numbers and apply kaiming initialization
        with torch.no_grad():
            fan = nn.init._calculate_correct_fan(self.output_layer.weight, 'fan_in')
            # this was set to relu before
            gain = 1.0
            std = gain / fan ** 0.5
            self.output_layer.weight.data = torch.normal(0, std, size=self.output_layer.weight.shape, 
                                                         generator=generator).to(self.device)       
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, 
                inputs: Dict[str, torch.Tensor]):
        # Ensure inputs are on the correct device
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.llm(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        # layer norm
        pred_hidden = self.llm.transformer.ln_f(pred_hidden)

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
                log_interval: int=100, patience: int=2, loss_fn_type='bce',
                need_ests: bool=False) -> None:
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
        self.patience = patience
        self.loss_fn_type = loss_fn_type
        self.need_ests = need_ests
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
        num_warmup_steps = self.num_warmup_steps * max_train_steps

        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Train loop
        best_loss = float('inf')
        best_epoch = 0
        # _ = self.eval_one_epoch()
        for epoch in range(self.num_train_epochs):
            self.model.train()
            self.train_one_epoch(epoch,)
            # if args.split < 1.0:
            self.model.eval()
            val_loss = self.eval_one_epoch()

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # current_lr = optimizer.param_groups[0]["lr"]
            self.save_checkpoint(epoch)

            if epochs_no_improve >= self.patience:
                self.logging(f'Early stopping at epoch {epoch}', 
                            self.model_dir + '/train.log')
                self.load_checkpoint(best_epoch)
                break

    def train_one_epoch(self, epoch,):
        self.optimizer.zero_grad()
        trainsize = len(self.train_dataloader)
        start = time.time()
        for i, batch in enumerate(self.train_dataloader):
            inputs, labels = batch
            if not self.need_ests:
                logits = self.model(inputs)
            else:
                logits = self.model(inputs, labels)
            if self.loss_fn_type in ['bce', 'mse']:
                loss = self.criterion(logits, labels.float())
            elif self.loss_fn_type == 'ce':
                loss = self.criterion(logits, labels)
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
        self.model.eval() # Set model to evaluate mode
        hits = 0
        total = 0
        total_loss = 0.0
        start = time.time()
        for i, batch in enumerate(self.val_dataloader):
            inputs, labels = batch
            # forward pass
            if not self.need_ests:
                preds = self.model(inputs)
            else:
                preds = self.model(inputs, labels)
            # print("shape of preds: ", preds.shape)
            # print("shape of labels: ", labels.shape)
            # calculate loss
            if self.loss_fn_type in ['bce', 'mse']:
                loss = self.criterion(preds, labels.float())
                if self.loss_fn_type == 'bce':
                    preds = torch.sigmoid(preds)
                else:
                    pass # preds should act like probabilities
            elif self.loss_fn_type == 'ce':
                loss = self.criterion(preds, labels)
                preds = torch.softmax(preds, dim=1)[:,1]
            total_loss += loss.item() * labels.size(0)
            preds = (preds > 0.5).long() # labels are long
            hits += sum(labels.view(-1) == preds.view(-1))
            # total += preds.size(0)
            total += preds.view(-1).size(0)
        elapsed_time = time.time() - start
        # print("Accuracy: {:.2f}".format(hits/total))
        avg_loss = total_loss / total
        acc = hits/total
        self.logging(f"Val acc local: {acc:.3f}", 
                    self.model_dir + '/train.log')
        self.logging(f"Val loss local: {avg_loss:.3f}", 
                    self.model_dir + '/train.log')
        self.logging(f"Time taken: {elapsed_time:.3f}",
                    self.model_dir + '/train.log')
        return avg_loss
    
    def load_checkpoint(self, best_epoch):
        fulloutput = os.path.join(self.model_dir, "checkpoint.{}".format(best_epoch))
        # check if the model exists
        if not os.path.exists(fulloutput):
            raise ValueError(f"Model checkpoint {fulloutput} does not exist")
        # pt_out = torch.load(f'{fulloutput}/pytorch_model.pt')
        self.model.load_state_dict(torch.load(f'{fulloutput}/pytorch_model.pt'), strict=False)
        self.model.tokenizer = AutoTokenizer.from_pretrained(fulloutput)

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
    
class PEWNetwork(nn.Module):
    def __init__(
        self,
        num_workers: int,
        model_path: str,
        seed: int,
        hidden_size_ow: int = 100,
        dropout_prob: float = 0.1,
        cache_dir: str = "scratch/cache",
        n_unfreeze: int = 0,
        loss_fn_type: str = 'mse',
    ):
        super(PEWNetwork, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.num_workers = num_workers
        self.dropout_prob = dropout_prob
        assert loss_fn_type in ['mse','ce'], "Loss function type must be either 'mse' or 'ce'"
        self.loss_fn_type = loss_fn_type

        # Determine device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained language model and move to the correct device
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir
        ).to(self.device)
        self.n_unfreeze = n_unfreeze
        for param in self.llm.parameters():
            param.requires_grad = False

        # Separate the frozen and unfrozen parts
        self.frozen_part = self.llm.transformer.h[:-n_unfreeze]
        self.unfrozen_part = self.llm.transformer.h[-n_unfreeze:]

        self.hidden_size_ow = hidden_size_ow
        for i in range(self.num_workers):
            setattr(self, f"lm_unfrozen_{i+1}", copy.deepcopy(self.unfrozen_part).to(self.device))
            setattr(self, f"ow_l1_{i+1}", nn.Linear(num_workers-1, hidden_size_ow).to(self.device))
            setattr(self, f"ow_l2_{i+1}", nn.Linear(self.llm.config.hidden_size + hidden_size_ow, 1).to(self.device))

        # Initialize weights
        self._initialize_weights()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _initialize_weights(self):
        generator = torch.Generator()  # Ensure generator is on the correct device
        generator.manual_seed(self.seed)
        # Manually generate random numbers and apply kaiming initialization
        for i in range(self.num_workers):
            with torch.no_grad():
                fan = nn.init._calculate_correct_fan(getattr(self, f"ow_l1_{i+1}").weight, 'fan_in')
                gain = nn.init.calculate_gain('relu')
                std = gain / fan ** 0.5
                getattr(self, f"ow_l1_{i+1}").weight.data = \
                    torch.normal(0, std, size=getattr(self, f"ow_l1_{i+1}").weight.shape, 
                                                                            generator=generator).to(self.device)       
            nn.init.zeros_(getattr(self, f"ow_l1_{i+1}").bias)
            with torch.no_grad():
                fan = nn.init._calculate_correct_fan(getattr(self, f"ow_l2_{i+1}").weight, 'fan_in')
                gain = 1.0
                std = gain / fan ** 0.5
                getattr(self, f"ow_l2_{i+1}").weight.data = \
                    torch.normal(0, std, size=getattr(self, f"ow_l2_{i+1}").weight.shape, 
                                                                            generator=generator).to(self.device)

    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                ests: torch.tensor,
                predict_gt: bool=False) -> torch.Tensor:
        # Ensure inputs are on the correct device
        attention_mask = inputs["attention_mask"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        # Check the shapes of input_ids and attention_mask
        assert input_ids.shape == attention_mask.shape, \
            f"input_ids shape {input_ids.shape} does not match attention_mask shape {attention_mask.shape}"
        # Get token embeddings (wte) and positional embeddings (wpe)
        token_embeddings = self.llm.transformer.wte(input_ids)  # Word/Token Embeddings
        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=self.device).unsqueeze(0)
        position_embeddings = self.llm.transformer.wpe(position_ids)  # Positional Embeddings
        # Combine token and positional embeddings
        hidden_states = token_embeddings + position_embeddings
        # ChatGPT says applying dropout is standard practice even though this is from the frozen part
        hidden_states = self.llm.transformer.drop(hidden_states)

        # Prepare attention mask for transformer blocks
        # See forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
        # specifically, lines 815 and 823
        # TODO: this might not work for all models
        ext_attention_mask = attention_mask[:, None, None, :].to(device=self.device).float()
        ext_attention_mask = (1.0 - ext_attention_mask) * torch.finfo(ext_attention_mask.dtype).min

        for layer in self.frozen_part:
            hidden_states = layer(hidden_states, attention_mask=ext_attention_mask)[0]

        preds = []
        for i in range(self.num_workers):
            branch_hidden_states = hidden_states.clone()
            for layer in getattr(self, f"lm_unfrozen_{i+1}"):
                branch_hidden_states = layer(branch_hidden_states, attention_mask=ext_attention_mask)[0]
            # apply layer norm -- see forward function in transformers.models.gpt2.modeling_gpt2.GPT2Model
            branch_hidden_states = self.llm.transformer.ln_f(branch_hidden_states)

            # Extract the hidden states at the positions specified by 'insizes'
            insizes = attention_mask.sum(dim=-1) - 1
            pred_hidden = branch_hidden_states[torch.arange(insizes.size(0)), insizes]
            # copy_outputs.append(pred_hidden)

            other_ids = torch.arange(self.num_workers)[torch.arange(self.num_workers) != i]
            ow_ests = ests[:,other_ids].clone().float()
            ow_op = getattr(self, f"ow_l1_{i+1}")(ow_ests)
            ow_op = torch.relu(ow_op)
            # apply dropout
            ow_op = nn.functional.dropout(ow_op, p=self.dropout_prob, training=self.training)
            pred_hidden = torch.cat((pred_hidden, ow_op), dim=1)
            pred_hidden = getattr(self, f"ow_l2_{i+1}")(pred_hidden)
            preds.append(pred_hidden)
        preds = torch.cat(preds, dim=1)
        if self.loss_fn_type == 'ce':
            # need preds to be of shape (batch_size, 2, num_workers)
            # concatenate preds with -preds
            if not predict_gt:
                preds = preds.unsqueeze(1)
                preds = torch.cat((preds, -preds), dim=1)
            else:
                # convert to probabilities
                preds = torch.sigmoid(preds)
                # avg over workers
                preds = torch.mean(preds, dim=1)
        elif self.loss_fn_type == 'mse' and predict_gt:
            # preds are probabilities already
            preds = torch.mean(preds, dim=1)
        return preds

class CrowdLayerNN(nn.Module):
    def __init__(
        self,
        model_path: str,
        seed: int,
        num_workers: int,
        dropout_prob: float = 0.1,
        cache_dir: str = "scratch/cache",
        no_freeze_lm: bool = True,
        n_unfreeze: int = 0,
        hidden_size: int = 100,
    ):
        super(CrowdLayerNN, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.num_workers = num_workers
        self.dropout_prob = dropout_prob

        # Determine device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained language model and move to the correct device
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir
        ).to(self.device)
        self.no_freeze_lm = no_freeze_lm
        self.n_unfreeze = n_unfreeze
        if not no_freeze_lm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            # Unfreeze the last n layers of GPT-2
            for layer in self.llm.transformer.h[-n_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # no freezing
            pass

        # Initialize additional layers and move them to the correct device
        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(self.llm.config.hidden_size, self.hidden_size).to(self.device)
        self.activation = nn.ReLU().to(self.device)
        self.output_layer = nn.Linear(hidden_size, 2).to(self.device)
        
        # Crowd layer
        for i in range(self.num_workers):
            setattr(self, "crowd_{}".format(i+1), torch.nn.Linear(2, 2, bias=False).to(self.device))

        # Initialize weights
        self._initialize_weights()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _initialize_weights(self):
        generator = torch.Generator()  # Ensure generator is on the correct device
        generator.manual_seed(self.seed)
        # TODO: gains are potentially incorrect.
        # Manually generate random numbers and apply kaiming initialization
        with torch.no_grad():
            fan = nn.init._calculate_correct_fan(self.hidden_layer.weight, 'fan_in')
            gain = nn.init.calculate_gain('relu')
            std = gain / fan ** 0.5
            self.hidden_layer.weight.data = torch.normal(0, std, size=self.hidden_layer.weight.shape, 
                                                         generator=generator).to(self.device)
        nn.init.zeros_(self.hidden_layer.bias)
        with torch.no_grad():
            fan = nn.init._calculate_correct_fan(self.output_layer.weight, 'fan_in')
            gain = 1.0
            std = gain / fan ** 0.5
            self.output_layer.weight.data = torch.normal(0, std, size=self.output_layer.weight.shape, 
                                                         generator=generator).to(self.device)       
        nn.init.zeros_(self.output_layer.bias)

        # crowd layer weights should be identity
        for i in range(self.num_workers):
            nn.init.eye_(getattr(self, "crowd_{}".format(i+1)).weight)

    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                predict_gt: bool = False):
        # Ensure inputs are on the correct device
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.llm(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        # apply layer norm
        pred_hidden = self.llm.transformer.ln_f(pred_hidden)

        # Apply deterministic dropout
        if self.training:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            dropout_mask = (torch.rand(pred_hidden.shape, generator=generator) > self.dropout_prob).float().to(pred_hidden.device)
            pred_hidden = pred_hidden * dropout_mask / (1 - self.dropout_prob)

        pred_hidden = self.hidden_layer(pred_hidden)
        pred_hidden = self.activation(pred_hidden)
        pred_hidden = self.output_layer(pred_hidden)
        # apply softmax
        pred_hidden = torch.softmax(pred_hidden, dim=-1)
        if predict_gt:
            return pred_hidden[:, 1]
        # crowd layer
        # logits = self.crowd_layer(pred_hidden)
        crowd_acts = []
        for i in range(self.num_workers):
            crowd_acts.append(getattr(self, "crowd_{}".format(i+1))(pred_hidden))
        crowd_acts = torch.stack(crowd_acts, dim=1)
        crowd_acts = crowd_acts.transpose(-1, -2)
        return crowd_acts
