import os

from transformers import AutoTokenizer
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from worker_agg import LMplusOneLayer
import worker_agg

def get_data(cfg, split_type='train', with_gt=False):
    data_constructor = worker_agg.__dict__[cfg.data_loader.name]
    data_dict = OmegaConf.to_container(cfg.data_loader.params, resolve=True, throw_on_missing=True)
    if split_type == 'val':
        data_dict['evalmode'] = True
    data_dict['with_gt'] = with_gt
    data = data_constructor(**data_dict)
    return data

def eval_model(model, dataloader):
    hits = 0
    total = 0
    for i, batch in enumerate(tqdm(dataloader)):
        inputs, ests, labels = batch
        logits = model(inputs)
        preds = (logits > 0).int()
        hits += sum(labels.view(-1) == preds.view(-1))
        total += preds.size(0)
    acc = hits/total
    return acc
    # print("Accuracy: {:.2f}".format(hits/total))

def load_checkpoint(model, model_dir, epoch):
    fulloutput = os.path.join(model_dir, "checkpoint.{}".format(epoch))
    # check if the model exists
    if not os.path.exists(fulloutput):
        raise ValueError(f"Model checkpoint {fulloutput} does not exist")
    # pt_out = torch.load(f'{fulloutput}/pytorch_model.pt')
    model.load_state_dict(torch.load(f'{fulloutput}/pytorch_model.pt'), strict=False)
    model.tokenizer = AutoTokenizer.from_pretrained(fulloutput)

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    model = LMplusOneLayer(
        model_path="gpt2",
        seed=69420,
    )
    # model_dir = "/home/akagr/Crowdsourcing-1/exp/lm_gt/2024-08-12_21-29-32"
    # epoch = 4
    # model_dir = "/home/akagr/Crowdsourcing-1/exp/lm_gt/2024-08-12_15-45-21"
    # epoch = 2
    model_dir = cfg.policy.params.model_dir
    epoch = cfg.eval.epoch
    load_checkpoint(model, model_dir, epoch)
    for split_type in ['train', 'val']:
        data = get_data(cfg, split_type=split_type, with_gt=True)
        dataloader = DataLoader(
                    data,
                    batch_size=cfg.policy.params.batch_size,
                    shuffle=False,
                    collate_fn=data.collate_fn,
                )
        acc = eval_model(model, dataloader)
        print(f"{split_type} accuracy: {acc}")

if __name__ == "__main__":
    main()