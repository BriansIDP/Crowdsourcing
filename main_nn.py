import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import worker_aggregation

def get_data(cfg, split_type='train', with_gt=False):
    data_constructor = worker_aggregation.__dict__[cfg.data_loader.name]
    data_dict = OmegaConf.to_container(cfg.data_gen.params, resolve=True, throw_on_missing=True)
    if split_type == 'val':
        data_dict['evalmode'] = True
    data_dict['with_gt'] = with_gt
    data = data_constructor(**data_dict)
    return data

def get_policy(cfg, ):
    policy_constructor = worker_aggregation.__dict__[cfg.policy.name]
    policy = policy_constructor(**cfg.policy.params)
    return policy

def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

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
    # with open(os.path.join(cfg.main.model_dir, 'model_config.json'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)
    policy = get_policy(cfg)
    if policy.name == 'LMGroundTruth':
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
