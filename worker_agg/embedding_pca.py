from pathlib import Path
import json

import numpy as np
import numpy.linalg as nl
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    # embeddings = np.load('dev/embeddings.npy')
    embeddings = np.load('data/gpt2_embeddings.npy')
    path = Path('data/halueval_dialogue.json')
    with open(path) as fin:
        all_data = json.load(fin)
    evidence_llms = ['llama3', 'beluga', 'mistral', 'zephyr', 'starling']
    outcomes = []
    binary_ests = []
    for datum in all_data:
        outcomes.append(0 if datum['ref'] == 'yes' else 1)
        est_vec = []
        for llm in evidence_llms:
            est = float(datum[llm][0]<0.5)
            est_vec.append(est)
        binary_ests.append(est_vec)
    outcomes = np.array(outcomes).flatten()
    binary_ests = np.array(binary_ests)
    assert outcomes.shape[0] == binary_ests.shape[0]
    assert outcomes.shape[0] == embeddings.shape[0]
    num_samples = outcomes.shape[0]
    num_models = binary_ests.shape[1]
    
    rng = np.random.default_rng(seed=42069)
    perm = rng.permutation(np.arange(len(outcomes)))
    embeddings = embeddings[perm]
    outcomes = outcomes[perm]
    binary_ests = binary_ests[perm, :]
    assert outcomes.shape == (num_samples,)
    assert binary_ests.shape == (num_samples, num_models)
    
    # print accuracy of each model
    for i in range(len(evidence_llms)):
        hits = 0
        for j in range(len(outcomes)):
            if binary_ests[j][i] == outcomes[j]:
                hits += 1
        print(f'{evidence_llms[i]} accuracy: {hits/len(outcomes)}')

    # split data
    train_size = int(0.6 * len(outcomes))
    val_size = int(0.2 * len(outcomes))
    test_size = len(outcomes) - train_size - val_size
    embed_train = embeddings[:train_size]
    embed_val = embeddings[train_size:train_size+val_size]
    embed_test = embeddings[train_size+val_size:]
    outcomes_train = outcomes[:train_size]
    outcomes_val = outcomes[train_size:train_size+val_size]
    outcomes_test = outcomes[train_size+val_size:]
    binary_ests_train = binary_ests[:train_size]
    binary_ests_val = binary_ests[train_size:train_size+val_size]
    binary_ests_test = binary_ests[train_size+val_size:]

    scaler = StandardScaler()
    embed_train = scaler.fit_transform(embed_train)
    embed_val = scaler.transform(embed_val)
    embed_test = scaler.transform(embed_test)
    pca = PCA()
    embed_train_pca = pca.fit_transform(embed_train)
    embed_val_pca = pca.transform(embed_val)
    embed_test_pca = pca.transform(embed_test)

    # explained_variance = np.cumsum(pca.explained_variance_ratio_)
    # num_components = np.where(explained_variance >= 0.9)[0][0] + 1
    # context_train = embed_train_pca[:, :num_components]
    # context_val = embed_val_pca[:, :num_components]
    # context_test = embed_test_pca[:, :num_components]

    # # Save context, outcomes, and binary_ests in a single .npz file
    # np.savez('data/bert_embed_90p_pca.npz',
    #         context_train=context_train,
    #         context_val=context_val,
    #         context_test=context_test,
    #         outcomes_train=outcomes_train,
    #         outcomes_val=outcomes_val,
    #         outcomes_test=outcomes_test,
    #         ests_train=binary_ests_train,
    #         ests_val=binary_ests_val,
    #         ests_test=binary_ests_test)

    # use the entire embedding as context
    context_train = embed_train
    context_val = embed_val
    context_test = embed_test
    # Save context, outcomes, and binary_ests in a single .npz file
    np.savez('data/gpt2_embed.npz',
            context_train=context_train,
            context_val=context_val,
            context_test=context_test,
            outcomes_train=outcomes_train,
            outcomes_val=outcomes_val,
            outcomes_test=outcomes_test,
            ests_train=binary_ests_train,
            ests_val=binary_ests_val,
            ests_test=binary_ests_test)


if __name__ == '__main__':
    main()