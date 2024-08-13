import numpy as np
# import matplotlib.pyplot as plt
import hydra

import worker_agg

def get_data(cfg):
    print(cfg.data_loader.name)
    data_constructor = worker_agg.__dict__[cfg.data_loader.name]
    out = data_constructor(**cfg.data_loader.params).get_data()
    return out

def get_data_val(cfg):
    data_constructor = worker_agg.__dict__[cfg.data_loader.name]
    out = data_constructor(**cfg.data_loader.params).get_data(split_type='val')
    return out

def get_policy(cfg, context_len=None):
    policy_constructor = worker_agg.__dict__[cfg.policy.name]
    neural_net_constructor = worker_agg.__dict__[cfg.neural_net.name]
    num_workers = len(cfg.data_loader.params.model_list)
    if 'needs_context' in cfg.policy:
        if cfg.policy.needs_context:
            assert context_len is not None
            if cfg.policy.name in ['EMNeuralNetBinary','NeuralNetMajVote']:
                def neural_net_cons():
                    return neural_net_constructor(**cfg.neural_net.params,
                                                    input_size=context_len)
                policy = policy_constructor(**cfg.policy.params, num_workers=num_workers,
                                            context_len=context_len, 
                                            neural_net_cons=neural_net_cons)
            else:
                policy = policy_constructor(**cfg.policy.params, num_workers=num_workers,
                                            context_len=context_len)
        else:
            policy = policy_constructor(**cfg.policy.params, num_workers=num_workers)
    else:
        if cfg.policy.name == 'AvgSSLPreds':
            neural_nets = [neural_net_constructor(input_size=num_workers-1, 
                                    **cfg.neural_net.params) for _ in range(num_workers)]
            policy = policy_constructor(**cfg.policy.params, num_workers=num_workers,
                                        neural_nets=neural_nets, 
                                        use_joblib_seeds=cfg.main.use_joblib_seeds)
        else: policy = policy_constructor(**cfg.policy.params, num_workers=num_workers)
    return policy

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    out = get_data(cfg)
    if len(out) == 2:
        ests, outcomes = out
        policy = get_policy(cfg)
        policy.fit(ests)
        group_ests = policy.predict(ests)
    elif len(out) == 3:
        contexts, ests, outcomes = out
        if 'needs_context' in cfg.policy:
            if cfg.policy.needs_context:
                policy = get_policy(cfg, contexts.shape[1])
                if 'ground_truth' in cfg.policy:
                    if cfg.policy.ground_truth:
                        policy.fit(contexts=contexts, ests=ests, outcomes=outcomes)
                    else:
                        policy.fit(contexts=contexts, ests=ests)
                else:
                    policy.fit(contexts=contexts, ests=ests)
                group_ests = policy.predict(contexts=contexts, ests=ests)
            else:
                policy = get_policy(cfg)
                policy.fit(ests)
                group_ests = policy.predict(ests)
        else:
            policy = get_policy(cfg)
            policy.fit(ests)
            group_ests = policy.predict(ests)
    else:
        raise ValueError("Data loader must return either 2 or 3 outputs")
    # group_ests_mv = np.mean(ests, axis=1)>0.5
    accuracy = np.mean(group_ests == outcomes)
    print(f"Accuracy: {accuracy:.3f}")

    if cfg.data_loader.name in ["HaluDialBertPCA", "HaluDialEmbed"]:
        out = get_data_val(cfg)
        if len(out) == 2:
            ests, outcomes = out
            group_ests = policy.predict(ests)
        elif len(out) == 3:
            contexts, ests, outcomes = out
            if 'needs_context' in cfg.policy:
                if cfg.policy.needs_context:
                    group_ests = policy.predict(contexts=contexts, ests=ests)
                else:
                    group_ests = policy.predict(ests)
            else:
                group_ests = policy.predict(ests)
        accuracy = np.mean(group_ests == outcomes)
        print(f"Validation Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()