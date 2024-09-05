import os
from datetime import datetime
import pytz
from typing import Union
import copy

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

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

def get_model_dir(cfg):
    now = datetime.now()
    # Convert to Pacific Time
    pacific_tz = pytz.timezone('US/Pacific')
    pacific_time = now.astimezone(pacific_tz)
    # Format the date-time string
    dt_string = pacific_time.strftime("%Y-%m-%d_%H-%M-%S")
    top_model_dir = os.path.join(cfg.policy.params.model_dir, dt_string)
    if os.path.exists(top_model_dir):
        raise ValueError(f"Directory {top_model_dir} already exists")
    os.makedirs(top_model_dir)
    return top_model_dir

def get_policy(cfg, top_model_dir, fold: Union[int, None]=None):
    # create model dir
    # top has policy and date
    # sub has fold number
    # return multiple models

    policy_constructor = worker_agg.__dict__[cfg.policy.name]
    model_constructor = worker_agg.__dict__[cfg.neural_net.name]
    policy_dict = copy.deepcopy(OmegaConf.to_container(cfg.policy.params, 
                                                       resolve=True, 
                                                       throw_on_missing=True))
    if fold is None:
        policy_dict['model_dir'] = top_model_dir
    else:
        policy_dict['model_dir'] = os.path.join(top_model_dir, f"fold_{fold}")
    if os.path.exists(policy_dict['model_dir']):
        raise ValueError(f"Directory {policy_dict['model_dir']} already exists")
    os.makedirs(policy_dict['model_dir'])
    if cfg.neural_net.name=='CombinedModel' and cfg.policy.name=='GTAsFeature':
        num_workers = len(cfg.data_loader.params.evidence_llm)
        model = model_constructor(**cfg.neural_net.params,
                                    num_workers=1)
    elif cfg.neural_net.name=='CombinedModel' and cfg.policy.name=='AvgSSLPredsSepLMs':
        num_workers = len(cfg.data_loader.params.evidence_llm)
        models = [model_constructor(**cfg.neural_net.params,
                                    num_workers=num_workers-1) for _ in range(num_workers)]
        policy = policy_constructor(**policy_dict, models=models)
        for i, model in enumerate(models):
            model_dir = os.path.join(policy_dict['model_dir'], f"model_{i}")
            os.makedirs(model_dir)
        return policy, policy_dict['model_dir']
    elif cfg.neural_net.name=='LMplusOneLayer' and cfg.policy.name=='PEWNoSSLSepLMs':
        num_workers = len(cfg.data_loader.params.evidence_llm)
        models = [model_constructor(**cfg.neural_net.params,) for _ in range(num_workers)]
        policy = policy_constructor(**policy_dict, models=models)
        for i, model in enumerate(models):
            model_dir = os.path.join(policy_dict['model_dir'], f"model_{i}")
            os.makedirs(model_dir)
        return policy, policy_dict['model_dir']
    elif cfg.neural_net.name in ['CrowdLayerNN', 'MultiHeadNet', 'CombinedModel']:
        num_workers = len(cfg.data_loader.params.evidence_llm)
        model = model_constructor(**cfg.neural_net.params,
                                    num_workers=num_workers)
        if cfg.policy.name in ['AvgSSLPredsLM','PEWNoSSL']:
            policy = policy_constructor(**policy_dict, model=model, num_workers=num_workers)
            return policy, policy_dict['model_dir']
    else:
        model = model_constructor(**cfg.neural_net.params)
    policy = policy_constructor(**policy_dict, model=model)
    return policy, policy_dict['model_dir']

def eval_policy(cfg, policy, dataloader):
    all_labels = []
    all_preds = []
    all_probs = []
    for i, batch in enumerate(tqdm(dataloader)):
        inputs, ests, labels = batch
        if cfg.neural_net.name=='CombinedModel' and cfg.policy.name=='GTAsFeature':
            preds, probs = policy.predict(inputs, labels.float())
        else:
            preds, probs = policy.predict(inputs, ests)
        all_labels.append(labels.view(-1).detach().cpu().numpy())
        all_preds.append(preds.view(-1).detach().cpu().numpy())
        all_probs.append(probs.view(-1).detach().cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    return all_labels, all_preds, all_probs
    # print("Accuracy: {:.2f}".format(hits/total))

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    if not cfg.data_loader.params.cross_val:
        policy, model_dir = get_policy(cfg)
        # please dump the config file to the model_dir
        with open(os.path.join(model_dir, 'model_config.yaml'), 'w') as f:
            OmegaConf.save(cfg, f)
        if cfg.policy.name in ['LMGroundTruth', 'GTAsFeature']:
            train_data = get_data(cfg, split_type='train', with_gt=True)
            val_data = get_data(cfg, split_type='val', with_gt=True)
        else:
            train_data = get_data(cfg, split_type='train', with_gt=False)
            val_data = get_data(cfg, split_type='val', with_gt=False)
        policy.fit(train_data, val_data)
        for split_type in ['train', 'val']:
            data = get_data(cfg, split_type=split_type, with_gt=True)
            dataloader = DataLoader(
                        data,
                        batch_size=policy.batch_size,
                        shuffle=False,
                        collate_fn=data.collate_fn,
                    )
            labels, preds, probs = eval_policy(cfg, policy, dataloader)
            acc = np.mean(labels == preds)
            f1 = f1_score(labels, preds)
            roc_auc = roc_auc_score(labels, probs)
            print(f"{split_type} accuracy: {acc}")
            print(f"{split_type} f1: {f1}")
            print(f"{split_type} roc_auc: {roc_auc}")
    else:
        top_model_dir = get_model_dir(cfg)
        labels = []
        preds = []
        probs = []
        for fold in range(cfg.data_loader.params.num_folds):
            policy, model_dir = get_policy(cfg, top_model_dir, fold)
            # please dump the config file to the model_dir
            with open(os.path.join(model_dir, 'model_config.yaml'), 'w') as f:
                OmegaConf.save(cfg, f)
            if cfg.policy.name in ['LMGroundTruth', 'GTAsFeature']:
                train_data = get_data(cfg, split_type='train', with_gt=True, fold=fold)
                val_data = get_data(cfg, split_type='val', with_gt=True, fold=fold)
            else:
                train_data = get_data(cfg, split_type='train', with_gt=False, fold=fold)
                val_data = get_data(cfg, split_type='val', with_gt=False, fold=fold)
            policy.fit(train_data, val_data)
            for split_type in ['train', 'val']:
                data = get_data(cfg, split_type=split_type, with_gt=True, fold=fold)
                dataloader = DataLoader(
                            data,
                            batch_size=policy.batch_size,
                            shuffle=False,
                            collate_fn=data.collate_fn,
                        )
                if split_type == 'train':
                    labels_, preds_, probs_ = eval_policy(cfg, policy, dataloader)
                elif split_type == 'val':
                    labels_, preds_, probs_ = eval_policy(cfg, policy, dataloader)
                    labels.append(labels_)
                    preds.append(preds_)
                    probs.append(probs_)
                acc = np.mean(labels_ == preds_)
                f1 = f1_score(labels_, preds_)
                roc_auc = roc_auc_score(labels_, probs_)
                print(f"{split_type} accuracy: {acc}")
                print(f"{split_type} f1: {f1}")
                print(f"{split_type} roc_auc: {roc_auc}")
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        # dump the labels and probs
        np.save(os.path.join(top_model_dir, 'labels.npy'), labels)
        np.save(os.path.join(top_model_dir, 'probs.npy'), probs)
        f1 = f1_score(labels, preds)
        acc = np.mean(labels == preds)
        roc_auc = roc_auc_score(labels, probs)
        print(f"Overall accuracy: {acc}")
        print(f"Overall f1: {f1}")
        print(f"Overall roc_auc: {roc_auc}")

if __name__ == "__main__":
    main()
