import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .lm_utils import FinetuneLM

class LMGroundTruth:
    def __init__(self, model,
                 num_workers: int,
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
        logits = self.model.predict(inputs,)
        preds = (logits>0).int().detach()
        return preds

class LMMajVote:
    def __init__(self, model,
                 num_workers: int,
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

    def collate_fn(self, batch):
        input_ids, ests = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        # ests = torch.stack(ests).to(self.device)
        mv_labels = torch.mode(torch.stack(ests), dim=-1).values.long().to(self.device)
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
        logits = self.model.predict(inputs,)
        preds = (logits>0).int().detach()
        return preds

# class LMMajVote(torch.nn.Module):
#     def __init__(
#         self,
#         model_path: str,
#         dropout_prob: float = 0.1,
#     ):
#         super().__init__()
#         self.llm = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             # cache_dir="/scratch/NeurowaveEval/leaderboard/bot/cache",  # Change to your local directory
#             cache_dir="scratch/cache",  # Change to your local directory
#         )
#         # for name, param in self.llm.named_parameters():
#         #     param.requires_grad = False
#         self.output_layer = torch.nn.Linear(self.llm.config.hidden_size, 1)
#         self.drop = torch.nn.Dropout(dropout_prob)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

#     def forward(self, 
#                 inputs: torch.Tensor, 
#                 estimates: torch.Tensor):
#         "estimates are assumed to be binary"
#         attention_mask = inputs["attention_mask"]
#         outputs = self.llm(
#             input_ids=inputs["input_ids"],
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             return_dict=True,
#         )
#         insizes = attention_mask.sum(dim=-1) - 1
#         pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
#         logits = self.output_layer(pred_hidden)
#         return logits
#         # prediction = torch.softmax(self.output_layer(pred_hidden), dim=-1)
#         # # apply majority vote to compute labels
#         # labels = torch.mode(estimates, dim=-1).values
#         # loss = torch.nn.functional.cross_entropy(prediction, labels.view(-1))
#         # return loss

#     def predict(self, 
#                 inputs: torch.Tensor, 
#                 estimates: torch.Tensor):
#         # estimates.requires_grad = True
#         attention_mask = inputs["attention_mask"]
#         outputs = self.llm(
#             input_ids=inputs["input_ids"],
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             return_dict=True,
#         )
#         insizes = attention_mask.sum(dim=-1) - 1
#         pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
#         prediction = torch.softmax(self.output_layer(pred_hidden), dim=-1)
#         # label is 1 if prediction is greater than 0.5
#         labels = (prediction[:, 1] > 0.5).int().detach()
#         return labels
#         # return prediction, pred_hidden
