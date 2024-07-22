import numpy as np
from sklearn.model_selection import train_test_split
import hydra

import worker_aggregation

def get_data(cfg):
    data_constructor = worker_aggregation.__dict__[cfg.data_loader.name]
    est_dict, outcomes = data_constructor.get_data(cfg.data_loader.params)
    model_list = list(est_dict.keys())
    all_ests = np.zeros((len(outcomes), len(model_list)))
    for i, model in enumerate(model_list):
        all_ests[:, i] = est_dict[model]

    # here is 60-20-20 split
    # TODO: make split percent configurable
    # seed_split = cfg.main.seed_split
    # ests_train_val, ests_test, outcome_train_val, outcome_test = \
    #     train_test_split(all_ests, outcomes, test_size=0.2, random_state=seed_split)

    # ests_train, ests_val, outcome_train, outcome_val = \
    #     train_test_split(ests_train_val, outcome_train_val, test_size=0.25, random_state=seed_split)
    ests_train = all_ests
    outcome_train = outcomes
    ests_val = all_ests
    outcome_val = outcomes
    ests_test = all_ests
    outcome_test = outcomes
    
    return {"train": (ests_train, outcome_train), "val": (ests_val, outcome_val), "test": (ests_test, outcome_test)}

def get_policy(cfg):
    policy_constructor = worker_aggregation.__dict__[cfg.policy.name]
    num_workers = len(cfg.data_loader.params.model_list)
    policy = policy_constructor(**cfg.policy.params, num_workers=num_workers)
    return policy

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    policy = get_policy(cfg)
    all_data = get_data(cfg)
    ests, outcomes = all_data[cfg.main.data_split]
    policy.fit(ests)
    group_ests = policy.predict(ests)
    accuracy = np.mean(group_ests == outcomes)
    if cfg.policy.name == "EM_GMM":
        print(cfg.policy.params.cov_mat_diag)
    print(f"Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()