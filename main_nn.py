import os
from datetime import datetime
import pytz

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import worker_agg

def get_data(cfg, split_type='train', with_gt=False):
    data_constructor = worker_agg.__dict__[cfg.data_loader.name]
    data_dict = OmegaConf.to_container(cfg.data_loader.params, resolve=True, throw_on_missing=True)
    if split_type == 'val':
        data_dict['evalmode'] = True
    data_dict['with_gt'] = with_gt
    data = data_constructor(**data_dict)
    return data

def get_policy(cfg, ):
    policy_constructor = worker_agg.__dict__[cfg.policy.name]
    model_constructor = worker_agg.__dict__[cfg.neural_net.name]
    now = datetime.now()
    # Convert to Pacific Time
    pacific_tz = pytz.timezone('US/Pacific')
    pacific_time = now.astimezone(pacific_tz)
    # Format the date-time string
    dt_string = pacific_time.strftime("%Y-%m-%d_%H-%M-%S")
    policy_dict = OmegaConf.to_container(cfg.policy.params, resolve=True, throw_on_missing=True)
    policy_dict['model_dir'] = os.path.join(cfg.policy.params.model_dir, dt_string)
    if os.path.exists(policy_dict['model_dir']):
        raise ValueError(f"Directory {policy_dict['model_dir']} already exists")
    os.makedirs(policy_dict['model_dir'])
    if cfg.neural_net.name in ['CrowdLayerNN', 'PEWNetwork']:
        num_workers = len(cfg.data_loader.params.model_list)
        model = model_constructor(**cfg.neural_net.params,
                                    num_workers=num_workers)
    policy = policy_constructor(**policy_dict, model=model)
    return policy, policy_dict['model_dir']

def eval_policy(policy, dataloader):
    hits = 0
    total = 0
    for i, batch in enumerate(dataloader):
        inputs, ests, labels = batch
        preds = policy.predict(inputs, ests)
        hits += sum(labels.view(-1) == preds.view(-1))
        total += preds.size(0)
    acc = hits/total
    return acc
    # print("Accuracy: {:.2f}".format(hits/total))

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    policy, model_dir = get_policy(cfg)
    # please dump the config file to the model_dir
    with open(os.path.join(model_dir, 'model_config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    if cfg.policy.name == 'LMGroundTruth':
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
        acc = eval_policy(policy, dataloader)
        print(f"{split_type} accuracy: {acc}")

if __name__ == "__main__":
    main()
