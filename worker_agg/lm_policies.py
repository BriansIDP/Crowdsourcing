import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .multihead_utils import FinetuneMultiHeadNet

class AvgSSLPredsLM:
    def __init__(self, model,
                 model_dir: str,
                 num_workers: int,
                 lr: float=0.001,
                 weight_decay: float=1e-5, 
                 gradient_accumulation_steps: int=1, 
                 num_warmup_steps: float=0.03,
                #  num_train_epochs: int=10,
                 max_grad_steps: int=5000,
                 lr_scheduler_type: str='cosine',
                 log_interval: int=100,
                 eval_interval: int=500,
                 batch_size: int=16,
                 probs: bool=False
                 ) -> None:
        self.model = model
        self.num_workers = num_workers
        self.model_dir = model_dir
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        # self.num_train_epochs = num_train_epochs
        self.max_grad_steps = max_grad_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.probs = probs

    def collate_fn(self, batch):
        input_ids, ests = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        ests = torch.stack(ests).to(self.device)
        if not self.probs:
            ests = ests.long()
            return (inputs, ests), ests
        else:
            return (inputs, ests), torch.cat((1.0-ests.unsqueeze(1), ests.unsqueeze(1)), dim=1)

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
                            #    num_train_epochs=self.num_train_epochs,
                               max_grad_steps=self.max_grad_steps,
                               lr_scheduler_type=self.lr_scheduler_type,
                               log_interval=self.log_interval,
                               eval_interval=self.eval_interval,
                               loss_fn_type=self.model.loss_fn_type,
                               probs=self.probs)
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
            probs = self.model((inputs, ests), predict_gt=True).detach().squeeze()
            preds = (probs>0.5).int().detach()
        return preds, probs