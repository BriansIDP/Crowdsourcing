import numpy as np
import hydra

import worker_aggregation

def get_data(cfg):
    data_constructor = worker_aggregation.__dict__[cfg.data_loader.name]
    # if "Syn" not in cfg.data_loader.name:
    est_dict, outcomes = data_constructor.get_data(**cfg.data_loader.params)
    model_list = list(est_dict.keys())
    ests = np.zeros((len(outcomes), len(model_list)))
    for i, model in enumerate(model_list):
        ests[:, i] = est_dict[model]
    return ests, outcomes
    # else:
    #     ests, features, outcomes = data_constructor.get_data(**cfg.data_loader.params)
    #     return ests, features, outcomes

def get_policy(cfg):
    policy_constructor = worker_aggregation.__dict__[cfg.policy.name]
    num_workers = len(cfg.data_loader.params.model_list)
    policy = policy_constructor(**cfg.policy.params, num_workers=num_workers)
    return policy

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    policy = get_policy(cfg)
    ests, outcomes = get_data(cfg)
    policy.fit(ests)
    group_ests = policy.predict(ests)
    accuracy = np.mean(group_ests == outcomes)
    print(f"Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()