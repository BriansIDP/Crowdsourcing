import os, sys
import json
import argparse

import numpy as np


np.random.seed(1)


def get_data(datapath, model_list):
    dataset = {}
    for model in model_list:
        hits = 0
        dataset[model] = []
        labels = []
        with open(os.path.join(datapath, "halueval_dialogue_{}.json".format(model))) as fin:
            modeldata = json.load(fin)[model]
        for datapiece in modeldata:
            true_label = 1 if datapiece["ref"] == "yes" else 0
            labels.append(true_label)
            dataset[model].append(datapiece["prob"])
            if datapiece["prob"][0] > datapiece["prob"][1] and true_label == 0:
                hits += 1
            elif datapiece["prob"][1] > datapiece["prob"][0] and true_label == 1:
                hits += 1
        print("{} Acc: {:.3f}".format(model, hits/len(dataset[model])))
    return dataset, labels


def get_artificial_data(mu_bar, sigma_bar, N, k):
    total = N // 2 * 2
    Z_pos = np.random.normal(0, 1, N//2)
    Z_pos = Z_pos * sigma_bar + mu_bar
    Z_neg = np.random.normal(0, 1, N//2)
    Z_neg = Z_neg * sigma_bar - mu_bar
    Z = np.concatenate([Z_pos, Z_neg], axis=-1)
    labels = Z < 0
    # labels = np.array([0 for _ in range(N//2)] + [1 for _ in range(N//2)])

    structure_N = 100
    C = np.random.normal(0, 1, (k, structure_N))
    modulate = np.array([[1/(n+1)**(p/10) for n in range(structure_N)] for p in range(k)])
    C = C * modulate
    X = np.random.normal(0, 1, (total, structure_N))
    delta = np.matmul(X, C.transpose())
    Y = Z[:, None] + delta
    return Y, labels, Z


def EM_orig(data, N, sigma_bar=2, rho_bar=0, c=0.1, M=10000, v_bar=1, mu_bar=0):
    Sigma_bar = np.identity(N) * sigma_bar + (np.ones((N, N)) - np.identity(N)) * rho_bar
    Sigma_hat = Sigma_bar
    epsilon = 1e-10
    m = 0
    T = data.shape[0]
    z_prev = 0 * np.ones(T)
    z_hat = 10000000 * np.ones(T)
    # z_hat = mu_bar * np.ones(T)
    # Start iteration
    while m < M and ((z_hat - z_prev) ** 2).mean() > epsilon:
        z_prev = z_hat
        Sigma_hat_inv = np.linalg.inv(Sigma_hat)
        # z_hat = np.matmul(data, Sigma_hat_inv).sum(axis=-1) / (1 + Sigma_hat_inv.sum())
        z_hat = np.matmul(data - mu_bar, np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bar
        v_hat = 1 / (v_bar + Sigma_hat_inv.sum())
        Y_cov = np.matmul((data-z_hat[:, None]).transpose(), data-z_hat[:, None])
        Y_cov += T * v_hat * np.ones((N, N))
        # Sigma_hat = (c * Sigma_bar + Y_cov) / (c + 2 * N + T + 1)
        Sigma_hat = Y_cov / T
        m += 1
    Z_hat = np.matmul(data - mu_bar, np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bar
    print("Done with {} steps".format(m))
    print("Precision Matrix")
    print(np.linalg.inv(Sigma_hat))
    weight = np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat).sum(axis=-1) * v_bar
    return 1 / (1 + np.exp(-Z_hat)), weight, Sigma_hat


def EM_bimodal(data, N, sigma_bar=2, rho_bar=0, c=0.1, M=10000, v_bar=1, mu_bar=0, assign="mean", labels=None):
    Sigma_bar = np.identity(N) * sigma_bar + (np.ones((N, N)) - np.identity(N)) * rho_bar
    Sigma_hat = Sigma_bar
    epsilon = 1e-10
    m = 0
    T = data.shape[0]
    z_prev = 0 * np.ones(T)
    z_hat = 10000000 * np.ones(T)
    # z_hat = mu_bar * np.ones(T)
    # Start iteration
    while m < M and ((z_hat - z_prev) ** 2).mean() > epsilon:
        # Get hard assignment
        if assign == "mean":
            mean_vec = data.mean(axis=-1)
        elif assign == "mode":
            mean_vec = ((data > 0) - 0.5).sum(axis=-1)
        elif assign == "gt":
            mean_vec = - labels + 0.5
        # if m > 1:
        #     mean_vec = z_hat
        pos_mask = mean_vec >= 0
        neg_mask = mean_vec < 0
        mu_bimodal = pos_mask * mu_bar - neg_mask * mu_bar

        z_prev = z_hat
        Sigma_hat_inv = np.linalg.inv(Sigma_hat)
        z_hat = np.matmul(data - mu_bimodal[:, None], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bimodal
        v_hat = 1 / (v_bar + Sigma_hat_inv.sum())
        Y_cov = np.matmul((data-z_hat[:, None]).transpose(), data-z_hat[:, None])
        Y_cov += T * v_hat * np.ones((N, N))
        Sigma_hat = Y_cov / T
        m += 1
    Z_hat = np.matmul(data - mu_bimodal[:, None], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bimodal
    print("Done with {} steps".format(m))
    print("Precision Matrix")
    print(np.linalg.inv(Sigma_hat))
    weight = np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat).sum(axis=-1) * v_bar
    return 1 / (1 + np.exp(-Z_hat)), weight, Sigma_hat


def main(args):
    model_list = ["llama3", "beluga", "mistral", "zephyr", "starling"]
    # model_list = ["zephyr", "llama3", "starling"]
    artificial = False
    v_bar_gen, mu_bar_gen = 1, 5
    if args.datapath == "artificial":
        data, labels, z_t = get_artificial_data(mu_bar_gen, v_bar_gen, 10000, 5)
        artificial = True
        np.save("outputs/gt.npy", z_t)
    else:
        dataset, labels = get_data(args.datapath, model_list)
        data_tensor = np.transpose(np.array([dataset[model] for model in model_list]), (1, 0, 2)) # .transpose(0, 1)
        labels = np.array(labels)

        # select one side
        # true_inds = np.where(labels == 1)
        # data_tensor = data_tensor[true_inds]
        # labels = labels[true_inds]

        # Direct averaging
        data = - np.log(1 / data_tensor[:, :, 0] - 1)

        # Moment matching
        # data_var = np.sqrt(np.var(data, axis=0)).max()
        # data = data / data_var

    data_mean = data.mean(axis=1)
    pos_center = ((data_mean >= 0) * data_mean).mean()
    neg_center = ((data_mean < 0) * data_mean).mean()
    # mu_bar = (pos_center - neg_center) / 2
    mu_bar = 5
    print("positive center: {}, negative center: {}".format(pos_center, neg_center))
    predicts = data_mean < 0
    hits = (labels == predicts).sum()
    print("Averaged Acc: {:.3f}".format(hits / len(labels)))
    data_mode = ((data > 0) - 0.5).sum(axis=-1)
    predicts = data_mode < 0
    hits = (labels == predicts).sum()
    print("Mode Acc: {:.3f}".format(hits / len(labels)))

    if args.algorithm == "em_orig":
        pred, weight, Sigma_hat = EM_orig(
            data,
            len(model_list),
            sigma_bar=2,
            rho_bar=0.0,
            c=0,
            M=10000,
            v_bar=1,
            mu_bar=0,
        )
    elif args.algorithm == "em_bimodal":
        pred, weight, Sigma_hat = EM_bimodal(
            data,
            len(model_list),
            sigma_bar=2,
            rho_bar=0.0,
            c=0,
            M=10000,
            v_bar=v_bar_gen if artificial else 5,
            mu_bar=mu_bar_gen if artificial else mu_bar,
            assign="gt",
            labels=labels,
        )
    print(weight)
    print(Sigma_hat)
    # pred = 1 / (1 + np.exp(-np.matmul(data, np.array([0.2, 0.8]))))
    predicts = pred < 0.5
    hits = (labels == predicts).sum()
    print("EM Acc: {:.3f}".format(hits / len(labels)))
    np.save("outputs/data.npy", data)


if __name__ == "__main__":
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--algorithm",
        type=str,
        default="em_orig",
        choices=["em_orig", "em_bimodal"],
        help="Aggregation method",
    )
    commandLineParser.add_argument(
        "--datapath",
        type=str,
        default="./data",
        help="Data path",
    )
    args = commandLineParser.parse_known_args()
    main(args[0])
