import os
from typing import Union

from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import hydra
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from worker_agg import LMplusOneLayer
import worker_agg

def get_data(cfg, split_type='train', with_gt: bool=False, 
             fold: Union[int, None]=None):
    data_constructor = worker_agg.__dict__[cfg.data_loader.name]
    data_dict = OmegaConf.to_container(cfg.data_loader.params, resolve=True, throw_on_missing=True)
    if split_type == 'val':
        data_dict['evalmode'] = True
    data_dict['with_gt'] = with_gt
    data_dict['fold'] = fold
    data = data_constructor(**data_dict)
    return data

def eval_model(cfg, model, dataloader):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    labels = []
    probs = []
    # ssl_preds = []
    for i, batch in enumerate(tqdm(dataloader)):
        inputs, ests, labels_ = batch
        labels.append(labels_.detach().squeeze().detach().cpu())
        if cfg.neural_net.name in ['MultiHeadNet']:
            probs_ = model((inputs, ests), predict_gt=True).squeeze().detach().cpu()
            # probs_, ssl_preds_ = model((inputs, ests), predict_gt=True, testing=True)
            # probs_ = probs_.squeeze().detach().cpu()
            # ssl_preds_ = ssl_preds_.detach().cpu()
            # ssl_preds.append(ssl_preds_)
        elif cfg.neural_net.name in ['CrowdLayerNN']:
            probs_ = model(inputs, predict_gt=True).squeeze().detach().cpu()
        elif cfg.neural_net.name in ['LMplusOneLayer']:
            probs_ = torch.sigmoid(model(inputs)).squeeze().detach().cpu()
        elif cfg.neural_net.name=='CombinedModel' and cfg.policy.name=='GTAsFeature':
            probs_ = torch.sigmoid(model((inputs, labels_.float()))).squeeze().detach().cpu()
        elif cfg.neural_net.name=='CombinedModel':
            probs_ = torch.sigmoid(model((inputs, ests))).squeeze().detach().cpu()
        probs.append(probs_)
        # preds.append((probs_ > 0.5).int())
        # hits += sum(labels.view(-1) == preds.view(-1))
        # total += preds.size(0)
    # probs = torch.cat(probs, dim=0)
    # all_labels = torch.cat(all_labels, dim=0)
    probs = torch.cat(probs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = (probs > 0.5).int()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # if cfg.neural_net.name in ['MultiHeadNet']:
    #     ssl_preds = torch.cat(ssl_preds, dim=0)
    #     breakpoint()
    #     return labels, preds, probs, ssl_preds
    return labels, preds, probs

# def eval_policy(cfg, policy, dataloader):
#     hits = 0
#     total = 0
#     probs = []
#     for i, batch in enumerate(tqdm(dataloader)):
#         inputs, ests, labels = batch
#         if cfg.neural_net.name=='CombinedModel' and cfg.policy.name=='GTAsFeature':
#             preds = policy.predict(inputs, labels.float())
#         else:
#             preds, _probs, _ = policy.predict(inputs, ests, testing=True)
#             probs.append(_probs.detach().cpu().numpy())
#         hits += sum(labels.view(-1) == preds.view(-1))
#         total += preds.size(0)
#     acc = hits/total
#     probs = np.concatenate(probs, axis=0)
#     return acc, probs


def create_collate_fn(i: int, num_workers: int):
    def collate_fn(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids, ests, labels = zip(*batch)
        labels = torch.tensor([l.item() for l in labels]).to(device).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
        attn_mask = input_ids != 0
        inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        other_ids = [j for j in range(num_workers) if j!=i]
        ests = torch.stack(ests).to(device)
        other_ests = ests[:, other_ids].float()
        # curr_est = ests[:, i:i+1].long()
        return inputs, other_ests, labels
    return collate_fn

def load_checkpoint(model, model_dir, epoch):
    fulloutput = os.path.join(model_dir, "checkpoint.{}".format(epoch))
    # check if the model exists
    if not os.path.exists(fulloutput):
        raise ValueError(f"Model checkpoint {fulloutput} does not exist")
    # pt_out = torch.load(f'{fulloutput}/pytorch_model.pt')
    model.load_state_dict(torch.load(f'{fulloutput}/pytorch_model.pt'), strict=False)
    # model.tokenizer = AutoTokenizer.from_pretrained(fulloutput)

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    model_constructor = worker_agg.__dict__[cfg.neural_net.name]
    if cfg.neural_net.name=='CombinedModel' and cfg.policy.name=='GTAsFeature':
        num_workers = len(cfg.data_loader.params.evidence_llm)
        model = model_constructor(**cfg.neural_net.params,
                                    num_workers=1)
    elif cfg.neural_net.name in ['CrowdLayerNN', 'MultiHeadNet', 'CombinedModel']:
        num_workers = len(cfg.data_loader.params.evidence_llm)
        model = model_constructor(**cfg.neural_net.params,
                                    num_workers=num_workers)
    else:
        model = model_constructor(**cfg.neural_net.params)
    if not cfg.data_loader.params.cross_val:
        model_dir = cfg.policy.params.model_dir
        epoch = cfg.eval.epoch
        load_checkpoint(model, model_dir, epoch)
        model.eval()
        probs_com = []
        ssl_preds_com = []
        for split_type in ['val', 'train']:
            data = get_data(cfg, split_type=split_type, with_gt=True, fold=0)
            dataloader = DataLoader(
                        data,
                        batch_size=cfg.policy.params.batch_size,
                        shuffle=False,
                        collate_fn=data.collate_fn,
                    )
            # if cfg.neural_net.name in ['MultiHeadNet']:
            #     labels, preds, probs, ssl_preds = eval_model(cfg, model, dataloader)
            #     ssl_preds_com.append(ssl_preds)
            # else:
            labels, preds, probs = eval_model(cfg, model, dataloader)
            probs_com.append(probs)
            acc = torch.mean((labels == preds).float()).item()
            print(f"{split_type} accuracy: {acc}")
        probs_com = torch.cat(probs_com, dim=0).cpu().detach().numpy()
        # np.save(f"{model_dir}/probs.npy", probs_com)
        # if cfg.neural_net.name in ['MultiHeadNet']:
        #     ssl_preds_com = torch.cat(ssl_preds_com, dim=0).cpu().detach().numpy()
        #     np.save(f"{model_dir}/ssl_preds.npy", ssl_preds_com)
    else:
        model_dir = cfg.policy.params.model_dir
        epochs = cfg.eval.epochs
        labels = []
        preds = []
        probs = []
        for fold in range(cfg.data_loader.params.nfolds):
            print(f"Fold {fold}")
            model_dir_fold = os.path.join(model_dir, f"fold_{fold}")
            load_checkpoint(model, model_dir_fold, epochs[fold])
            data = get_data(cfg, split_type='val', with_gt=True, fold=fold)
            dataloader = DataLoader(
                        data,
                        batch_size=cfg.policy.params.batch_size,
                        shuffle=False,
                        collate_fn=data.collate_fn,
                    )
            labels_, preds_, probs_ = eval_model(cfg, model, dataloader)
            labels.append(labels_.numpy())
            preds.append(preds_.numpy())
            probs.append(probs_.numpy())
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
        f1 = f1_score(labels, preds)
        acc = np.mean(labels == preds)
        roc_auc = roc_auc_score(labels, probs)
        print(f"Overall accuracy: {acc}")
        print(f"Overall f1: {f1}")
        print(f"Overall roc_auc: {roc_auc}")
        # dump the labels and probs
        np.save(os.path.join(model_dir, 'labels.npy'), labels)
        np.save(os.path.join(model_dir, 'probs.npy'), probs)

    # policy_constructor = worker_agg.__dict__[cfg.policy.name]
    # model_constructor = worker_agg.__dict__[cfg.neural_net.name]
    # num_workers = len(cfg.data_loader.params.evidence_llm)
    # model_dir = cfg.policy.params.model_dir
    # epochs = cfg.eval.epochs
    # probs_com = []
    # for split_type in ['train','val']:
    #     probs = []
    #     data = get_data(cfg, split_type=split_type, with_gt=True)
    #     for i in range(num_workers):
    #         model = model_constructor(**cfg.neural_net.params, num_workers=num_workers-1)
    #         model_dir_i = os.path.join(model_dir, f"model_{i}")
    #         load_checkpoint(model, model_dir_i, epochs[i])
    #         dataloader = DataLoader(
    #                     data,
    #                     batch_size=cfg.policy.params.batch_size,
    #                     shuffle=False,
    #                     collate_fn=create_collate_fn(i, num_workers),
    #                 )
    #         acc_i, probs_i, all_labels = eval_model(cfg, model, dataloader)
    #         print(probs_i.shape)
    #         print(f"{split_type} accuracy model {i}: {acc_i}")
    #         probs.append(probs_i.cpu().detach().numpy())
    #     probs = np.concatenate(probs, axis=1)
    #     print(probs.shape)
    #     probs = np.mean(probs, axis=1)
    #     probs_com += probs.tolist()
    #     pred_labels = (probs > 0.5).astype(int)
    #     acc = np.mean(pred_labels == all_labels.cpu().numpy())
    #     print(f"{split_type} accuracy: {acc}")
    # np.save(f"{model_dir}/probs.npy", probs_com)

if __name__ == "__main__":
    main()