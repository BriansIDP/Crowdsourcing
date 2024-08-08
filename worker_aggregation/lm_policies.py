import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class LMMajVote(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            # cache_dir="/scratch/NeurowaveEval/leaderboard/bot/cache",  # Change to your local directory
            cache_dir="scratch/cache",  # Change to your local directory
        )
        # for name, param in self.llm.named_parameters():
        #     param.requires_grad = False
        self.output_layer = torch.nn.Linear(self.llm.config.hidden_size, 2)
        self.activation = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout_prob)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, 
                inputs: torch.Tensor, 
                estimates: torch.Tensor):
        "estimates are assumed to be binary"
        attention_mask = inputs["attention_mask"]
        outputs = self.llm(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        prediction = torch.softmax(self.output_layer(pred_hidden), dim=-1)
        # apply majority vote to compute labels
        labels = torch.mode(estimates, dim=-1).values
        loss = torch.nn.functional.cross_entropy(prediction, labels.view(-1))
        return loss

    def predict(self, 
                inputs: torch.Tensor, 
                estimates: torch.Tensor):
        # estimates.requires_grad = True
        attention_mask = inputs["attention_mask"]
        outputs = self.llm(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        insizes = attention_mask.sum(dim=-1) - 1
        pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
        prediction = torch.softmax(self.output_layer(pred_hidden), dim=-1)
        # label is 1 if prediction is greater than 0.5
        labels = (prediction[:, 1] > 0.5).int().detach()
        return labels
        # return prediction, pred_hidden
