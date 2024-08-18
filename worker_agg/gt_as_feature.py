from typing import Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lm_utils import FinetuneLM

class CombinedModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        seed: int,
        num_workers: int,
        hidden_size: int, 
        dropout_prob: float = 0.1,
        cache_dir: str = "scratch/cache",
        no_freeze_lm: bool = True,
        n_unfreeze: int = 0,
    ):
        super().__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.dropout_prob = dropout_prob
        self.num_workers = num_workers

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
            if n_unfreeze > 0:
                for layer in self.llm.transformer.h[-n_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                pass # no unfreezing
        else:
            # no freezing
            pass

        # Initialize additional layers and move them to the correct device
        self.hidden_size = hidden_size
        self.com_layer = nn.Linear(self.llm.config.hidden_size + num_workers, hidden_size).to(self.device)
        self.output_layer = nn.Linear(hidden_size, 1).to(self.device)

        # Initialize weights
        self._initialize_weights()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _initialize_weights(self):
        generator = torch.Generator()  # Ensure generator is on the correct device
        generator.manual_seed(self.seed)
        # Manually generate random numbers and apply kaiming initialization
        with torch.no_grad():
            fan = nn.init._calculate_correct_fan(self.com_layer.weight, 'fan_in')
            gain = nn.init.calculate_gain('relu')
            std = gain / fan ** 0.5
            self.com_layer.weight.data = torch.normal(0, std, size=self.com_layer.weight.shape, 
                                                         generator=generator).to(self.device)       
        nn.init.zeros_(self.com_layer.bias)
        with torch.no_grad():
            fan = nn.init._calculate_correct_fan(self.output_layer.weight, 'fan_in')
            gain = 1.0
            std = gain / fan ** 0.5
            self.output_layer.weight.data = torch.normal(0, std, size=self.output_layer.weight.shape, 
                                                         generator=generator).to(self.device)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                ests: torch.Tensor,):
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
        # # dropout
        # pred_hidden = torch.dropout(pred_hidden, p=self.dropout_prob, train=self.training)

        # Concatenate the estimated values
        pred_hidden = torch.cat([pred_hidden, ests], dim=-1)
        pred_hidden = torch.relu(self.com_layer(pred_hidden))
        pred_hidden = torch.dropout(pred_hidden, p=self.dropout_prob, train=self.training)

        logits = self.output_layer(pred_hidden)
        return logits

class GTAsFeature:
    def __init__(self, model,
                #  num_workers: int, 
                 model_dir: str,
                 lr: float=0.001,
                 weight_decay: float=1e-5, 
                 gradient_accumulation_steps: int=1, 
                 num_warmup_steps: float=0.03,
                 num_train_epochs: int=10,
                 lr_scheduler_type: str='cosine',
                 log_interval: int=100,
                 batch_size: int=16,
                 ) -> None:
        self.model = model
        # self.num_workers = num_workers
        self.model_dir = model_dir
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.num_train_epochs = num_train_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(self, batch):
        input_ids, ests, outcomes = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        # ests = torch.stack(ests).to(self.device)
        outcomes = torch.stack(outcomes).to(self.device)
        return inputs, outcomes

    def fit(self, train_data, val_data):
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        finetuner = FinetuneLM(model=self.model,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               model_dir=self.model_dir,
                               lr=self.lr,
                               weight_decay=self.weight_decay,
                               gradient_accumulation_steps=self.gradient_accumulation_steps,
                               num_warmup_steps=self.num_warmup_steps,
                               num_train_epochs=self.num_train_epochs,
                               lr_scheduler_type=self.lr_scheduler_type,
                               log_interval=self.log_interval,
                               need_ests=True,
                               loss_fn_type='bce',)
        finetuner.run()

    def predict(self, inputs, outcomes):
        logits = self.model(inputs, outcomes)
        preds = (logits>0).int().detach()
        return preds
