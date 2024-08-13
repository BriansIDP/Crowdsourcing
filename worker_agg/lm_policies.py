import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .lm_utils import FinetuneLM

class LMGroundTruth:
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
                               log_interval=self.log_interval)
        finetuner.run()

    def predict(self, inputs, ests):
        logits = self.model(inputs,)
        preds = (logits>0).int().detach()
        return preds

class LMMajVote:
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
        input_ids, ests = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        # ests = torch.stack(ests).to(self.device)
        mv_labels = torch.mode(torch.stack(ests), dim=-1).values.long().to(self.device)
        mv_labels = mv_labels[:, None]
        return inputs, mv_labels

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
                               log_interval=self.log_interval)
        finetuner.run()

    def predict(self, inputs, ests):
        logits = self.model(inputs,)
        preds = (logits>0).int().detach()
        return preds
