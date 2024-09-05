import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os

from .lm_utils import FinetuneLM
from .multihead_utils import FinetuneMultiHeadNet

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
                 patience: int=2,
                 loss_fn_type='bce',
                 seed: int=42
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
        self.patience = patience
        self.loss_fn_type = loss_fn_type
        self.seed = seed

    def collate_fn(self, batch):
        input_ids, ests, outcomes = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        # ests = torch.stack(ests).to(self.device)
        outcomes = torch.stack(outcomes).to(self.device)
        if self.loss_fn_type == 'ce':
            outcomes = outcomes.long().squeeze()
        return inputs, outcomes

    def fit(self, train_data, val_data):
        generator = torch.Generator().manual_seed(self.seed)
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            generator=generator
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
                               patience=self.patience,
                               loss_fn_type=self.loss_fn_type)
        finetuner.run()

    def predict(self, inputs, ests):
        if self.loss_fn_type == 'bce':
            logits = self.model(inputs,)
            preds = (logits>0).int().detach()
        elif self.loss_fn_type == 'ce':
            logits = self.model(inputs,)
            preds = torch.argmax(logits, dim=-1)
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
                 patience: int=2
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
        self.patience = patience

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
                               log_interval=self.log_interval,
                               patience=self.patience)
        finetuner.run()

    def predict(self, inputs, ests):
        logits = self.model(inputs,)
        preds = (logits>0).int().detach()
        return preds

class CrowdLayerLM:
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
        ests = torch.stack(ests).to(self.device).long()
        return inputs, ests

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
                               loss_fn_type='ce')
        finetuner.run()

    def predict(self, inputs, ests):
        sigmoids = self.model(inputs, predict_gt=True)
        preds = (sigmoids>0.5).int().detach()
        return preds

class AvgSSLPredsLM:
    def __init__(self, model,
                 model_dir: str,
                 num_workers: int,
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
        self.num_workers = num_workers
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
        ests = torch.stack(ests).to(self.device).long()
        return (inputs, ests), ests

    def fit(self, train_data, val_data):
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_dataloader = DataLoader(
            val_data,
            # batch_size=self.batch_size,
            batch_size=8,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        finetuner = FinetuneMultiHeadNet(model=self.model,
                                num_workers=self.num_workers,
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
                               loss_fn_type=self.model.loss_fn_type,)
        finetuner.run()

    def predict(self, inputs, ests: torch.Tensor,
                predict_gt: bool=True):
        if not predict_gt:
            ssl_preds = self.model((inputs, ests), predict_gt=predict_gt)
            # ssl_preds are logits if loss_fn_type is 'ce'
            # else they are linear layer outputs
            return ssl_preds
        else:
            # preds are avg of linear layer outputs
            preds = self.model((inputs, ests), predict_gt=True)
            preds = (preds>0.5).int().detach()
        return preds

class PEWNoSSL:
    def __init__(self, model,
                 model_dir: str,
                 num_workers: int,
                 lr: float=0.001,
                 weight_decay: float=1e-5, 
                 gradient_accumulation_steps: int=1, 
                 num_warmup_steps: float=0.03,
                 num_train_epochs: int=10,
                 lr_scheduler_type: str='cosine',
                 log_interval: int=100,
                 batch_size: int=16,
                 patience: int=2
                 ) -> None:
        self.model = model
        self.num_workers = num_workers
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
        self.patience = patience

    def collate_fn(self, batch):
        input_ids, ests = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        ests = torch.stack(ests).to(self.device).long()
        return inputs, ests

    def fit(self, train_data, val_data):
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_dataloader = DataLoader(
            val_data,
            # batch_size=self.batch_size,
            batch_size=8,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        finetuner = FinetuneMultiHeadNet(model=self.model,
                                num_workers=self.num_workers,
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
                               loss_fn_type=self.model.loss_fn_type,
                               patience=self.patience)
        finetuner.run()

    def predict(self, inputs, ests: torch.Tensor,
                predict_gt: bool=True):
        if not predict_gt:
            ssl_preds = self.model(inputs, predict_gt=predict_gt)
            # ssl_preds are logits if loss_fn_type is 'ce'
            # else they are linear layer outputs
            return ssl_preds
        else:
            # preds are avg of linear layer outputs
            preds = self.model(inputs, predict_gt=True)
            preds = (preds>0.5).int().detach()
        return preds

class AvgSSLPredsSepLMs:
    def __init__(self, models,
                 model_dir: str,
                 lr: float=0.001,
                 weight_decay: float=1e-5, 
                 gradient_accumulation_steps: int=1, 
                 num_warmup_steps: float=0.03,
                 num_train_epochs: int=10,
                 lr_scheduler_type: str='cosine',
                 log_interval: int=100,
                 batch_size: int=16,
                 loss_fn_type: str='bce',
                 ) -> None:
        self.models = models
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
        self.loss_fn_type = loss_fn_type

    def create_collate_fn(self, i: int):
        def collate_fn(batch):
            input_ids, ests = zip(*batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
            attn_mask = input_ids != 0
            inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
            other_ids = [j for j in range(len(self.models)) if j!=i]
            ests = torch.stack(ests).to(self.device)
            other_ests = ests[:, other_ids].float()
            curr_est = ests[:, i:i+1].long()
            return (inputs, other_ests), curr_est
        return collate_fn

    def fit(self, train_data, val_data):
        for i in range(len(self.models)):
            print(f"Training model {i}")
            train_dataloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.create_collate_fn(i),
            )
            val_dataloader = DataLoader(
                val_data,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.create_collate_fn(i),
            )
            # os.makedirs(policy_dict['model_dir'])
            model_dir_i = os.path.join(self.model_dir, f"model_{i}")
            finetuner = FinetuneLM(model=self.models[i],
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                model_dir=model_dir_i,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                gradient_accumulation_steps=self.gradient_accumulation_steps,
                                num_warmup_steps=self.num_warmup_steps,
                                num_train_epochs=self.num_train_epochs,
                                lr_scheduler_type=self.lr_scheduler_type,
                                log_interval=self.log_interval,
                                loss_fn_type=self.loss_fn_type,)
            finetuner.run()

    def predict(self, inputs, ests: torch.Tensor, testing: bool=False):
        ssl_preds = []
        for i in range(len(self.models)):
            self.models[i].eval()
            other_ids = [j for j in range(len(self.models)) if j!=i]
            other_ests = ests[:, other_ids].float()
            if self.loss_fn_type == 'bce':
                ssl_preds.append(torch.sigmoid(self.models[i]((inputs, other_ests))))
            elif self.loss_fn_type == 'mse':
                ssl_preds.append(self.models[i]((inputs, other_ests)).view(-1))
            else:
                raise ValueError("loss_fn_type should be 'bce' or 'mse'")
        ssl_preds = torch.stack(ssl_preds).transpose(1, 0)
        # print(ssl_preds.shape)
        preds = torch.mean(ssl_preds, dim=1)
        labels = (preds>0.5).int().detach()
        if not testing:
            return labels
        else:
            return labels, preds, ssl_preds

class PEWNoSSLSepLMs:
    def __init__(self, models,
                 model_dir: str,
                 lr: float=0.001,
                 weight_decay: float=1e-5, 
                 gradient_accumulation_steps: int=1, 
                 num_warmup_steps: float=0.03,
                 num_train_epochs: int=10,
                 lr_scheduler_type: str='cosine',
                 log_interval: int=100,
                 batch_size: int=16,
                 loss_fn_type: str='bce',
                 ) -> None:
        self.models = models
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
        self.loss_fn_type = loss_fn_type

    def create_collate_fn(self, i: int):
        def collate_fn(batch):
            input_ids, ests = zip(*batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
            attn_mask = input_ids != 0
            inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
            ests = torch.stack(ests).to(self.device)
            curr_est = ests[:, i:i+1].long()
            return inputs, curr_est
        return collate_fn

    def fit(self, train_data, val_data):
        for i in range(len(self.models)):
            print(f"Training model {i}")
            train_dataloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.create_collate_fn(i),
            )
            val_dataloader = DataLoader(
                val_data,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.create_collate_fn(i),
            )
            # os.makedirs(policy_dict['model_dir'])
            model_dir_i = os.path.join(self.model_dir, f"model_{i}")
            finetuner = FinetuneLM(model=self.models[i],
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                model_dir=model_dir_i,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                gradient_accumulation_steps=self.gradient_accumulation_steps,
                                num_warmup_steps=self.num_warmup_steps,
                                num_train_epochs=self.num_train_epochs,
                                lr_scheduler_type=self.lr_scheduler_type,
                                log_interval=self.log_interval,
                                loss_fn_type=self.loss_fn_type,)
            finetuner.run()

    def predict(self, inputs, ests: torch.Tensor, testing: bool=False):
        ssl_preds = []
        for i in range(len(self.models)):
            self.models[i].eval()
            if self.loss_fn_type == 'bce':
                ssl_preds.append(torch.sigmoid(self.models[i](inputs)))
            elif self.loss_fn_type == 'mse':
                ssl_preds.append(self.models[i](inputs).view(-1))
            else:
                raise ValueError("loss_fn_type should be 'bce' or 'mse'")
        ssl_preds = torch.stack(ssl_preds).transpose(1, 0)
        # print(ssl_preds.shape)
        preds = torch.mean(ssl_preds, dim=1)
        labels = (preds>0.5).int().detach()
        if not testing:
            return labels
        else:
            return labels, preds, ssl_preds
