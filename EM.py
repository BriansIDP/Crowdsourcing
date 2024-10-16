import os, sys
import json
import argparse

import numpy as np
from scipy.stats import norm


np.random.seed(1)


def get_data(datapath, model_list):
    dataset = {}
    threshold = 0.5
    for model in model_list:
        hits = 0
        dataset[model] = []
        labels = []
        with open(os.path.join(datapath, "halueval_dialogue_{}.json".format(model))) as fin:
            modeldata = json.load(fin)[model]
        for datapiece in modeldata:
            true_label = 0 if datapiece["ref"] == "yes" else 1
            labels.append(true_label)
            dataset[model].append(datapiece["prob"])
            # if datapiece["prob"][0] > datapiece["prob"][1] and true_label == 0:
            if datapiece["prob"][0] > threshold and true_label == 0:
                hits += 1
            # elif datapiece["prob"][1] > datapiece["prob"][0] and true_label == 1:
            elif datapiece["prob"][0] < threshold and true_label == 1:
                hits += 1
        print("{} Acc: {:.3f}".format(model, hits/len(dataset[model])))
    return dataset, labels


def get_artificial_data(mu_bar, sigma_bar, mean_1, mean_2, cov_1, cov_2, N):
    total = N // 2 * 2
    Z_pos = np.random.normal(0, 1, total//2)
    Z_pos = Z_pos * sigma_bar + mu_bar
    Z_neg = np.random.normal(0, 1, total//2)
    Z_neg = Z_neg * sigma_bar - mu_bar
    Z = np.concatenate([Z_pos, Z_neg], axis=-1)
    labels = Z < 0
    # labels = np.array([0 for _ in range(N//2)] + [1 for _ in range(N//2)])

    # structure_N = 100
    # C = np.random.normal(0, 1, (k, structure_N))
    # modulate = np.array([[1/(n+1)**(p/10) for n in range(structure_N)] for p in range(k)])
    # C = C * modulate
    # X = np.random.normal(0, 1, (total, structure_N))
    # delta = np.matmul(X, C.transpose())
    delta_pos = np.random.multivariate_normal(mean_2, cov_2, total//2)
    delta_neg = np.random.multivariate_normal(mean_1, cov_1, total//2)
    delta = np.concatenate([delta_pos, delta_neg], axis=0)
    Y = Z[:, None] + delta
    return Y, labels, Z


def EM_Gmixture(data, sigma_bar=2, rho_bar=0, M=10000, p=0.5, mu_1_init=1, mu_2_init=-1):
    N = data.shape[1]
    Sigma_bar = np.identity(N) * sigma_bar + (np.ones((N, N)) - np.identity(N)) * rho_bar
    Sigma_hat_1 = Sigma_bar
    Sigma_hat_2 = Sigma_bar
    epsilon = 1e-10
    m = 0
    T = data.shape[0]
    q_1 = p * np.ones(T)
    q_2 = 1 - p * np.ones(T)
    # q_1_prev = 0 * np.ones(T)
    mu_1 = np.ones(N) * mu_1_init
    mu_2 = np.ones(N) * mu_2_init
    mu_2_prev = 0 * mu_2
    while m < M and np.max(np.abs((mu_2 - mu_2_prev))) > epsilon:
        q_1_prev = q_1
        mu_2_prev = mu_2
        # E-step
        det_Sigma_1 = np.linalg.det(Sigma_hat_1)
        det_Sigma_2 = np.linalg.det(Sigma_hat_2)
        pos_debias = data - mu_1[None, :]
        exp_pos = 1 / np.sqrt(det_Sigma_1) * np.exp(- 0.5 * np.sum(np.matmul(pos_debias, np.linalg.inv(Sigma_hat_1)) * pos_debias, axis=-1))
        neg_debias = data - mu_2[None, :]
        exp_neg = 1 / np.sqrt(det_Sigma_2) * np.exp(- 0.5 * np.sum(np.matmul(neg_debias, np.linalg.inv(Sigma_hat_2)) * neg_debias, axis=-1))
        q_1  = (exp_pos * p) / (exp_pos * p + exp_neg * (1 - p))
        # q_2 = (exp_neg * (1 - p)) / (exp_pos * p + exp_neg * (1 - p))
        q_2 = 1 - q_1

        # M-step
        mu_1 = (q_1[:, None] * data).sum(axis=0) / q_1.sum()
        mu_2 = (q_2[:, None] * data).sum(axis=0) / q_2.sum()
        Sigma_hat_1 = np.matmul((data - mu_1).transpose(), q_1[:, None] * (data - mu_1)) / q_1.sum()
        Sigma_hat_2 = np.matmul((data - mu_2).transpose(), q_2[:, None] * (data - mu_2)) / q_2.sum()
        m += 1

    # MLE assign
    print("Done with {} steps".format(m))
    det_Sigma_1 = np.linalg.det(Sigma_hat_1)
    det_Sigma_2 = np.linalg.det(Sigma_hat_2)
    pos_debias = data - mu_1[None, :]
    exp_pos = 1 / np.sqrt(det_Sigma_1) * np.exp(- 0.5 * np.sum(np.matmul(pos_debias, np.linalg.inv(Sigma_hat_1)) * pos_debias, axis=-1))
    neg_debias = data - mu_2[None, :]
    exp_neg = 1 / np.sqrt(det_Sigma_2) * np.exp(- 0.5 * np.sum(np.matmul(neg_debias, np.linalg.inv(Sigma_hat_2)) * neg_debias, axis=-1))
    q_1  = (exp_pos * p) / (exp_pos * p + exp_neg * (1 - p))
    return q_1 < 0.5, Sigma_hat_1, Sigma_hat_2, mu_1, mu_2


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
    # print("Precision Matrix")
    # print(np.linalg.inv(Sigma_hat))
    weight = np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat).sum(axis=-1) * v_bar
    return Z_hat, weight, Sigma_hat


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
        if assign == "likelihood":
            Z_hat_pos = np.matmul(data - mu_bar, np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bar
            Z_hat_neg = np.matmul(data + mu_bar, np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar - mu_bar
            pos_mean_dev = data - Z_hat_pos[:, None]
            exp_pos = - np.sum(np.matmul(pos_mean_dev, np.linalg.inv(Sigma_hat)) * pos_mean_dev, axis=-1)
            neg_mean_dev = data - Z_hat_neg[:, None]
            exp_neg = - np.sum(np.matmul(neg_mean_dev, np.linalg.inv(Sigma_hat)) * neg_mean_dev, axis=-1)
            pos_mask = exp_pos >= exp_neg
            neg_mask = exp_pos < exp_neg
        else:
            pos_mask = mean_vec >= 0
            neg_mask = mean_vec < 0
        mu_bimodal = pos_mask * mu_bar - neg_mask * mu_bar

        z_prev = z_hat
        Sigma_hat_inv = np.linalg.inv(Sigma_hat)
        z_hat = np.matmul(data - mu_bimodal[:, None], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bimodal
        v_hat = 1 / (1/v_bar + Sigma_hat_inv.sum())
        Y_cov = np.matmul((data-z_hat[:, None]).transpose(), data-z_hat[:, None])
        Y_cov += T * v_hat * np.ones((N, N))
        Sigma_hat = Y_cov / T
        m += 1
    Z_hat = np.matmul(data - mu_bimodal[:, None], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bimodal
    # # check how many signs of mean_vec and Z_hat are different
    # print("Mean and Z_hat sign difference: {}".format(((mean_vec >= 0) != (Z_hat >= 0)).sum()))
    print("Done with {} steps".format(m))
    Sigma_hat_inv = np.linalg.inv(Sigma_hat)
    v_hat = 1 / (1/v_bar + Sigma_hat_inv.sum())

    # Inference
    Z_hat_pos = np.matmul(data - mu_bar, np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bar
    Z_hat_neg = np.matmul(data + mu_bar, np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar - mu_bar
    pos_mean_dev = data - Z_hat_pos[:, None]
    exp_pos = np.exp(-0.5 * np.sum(np.matmul(pos_mean_dev, np.linalg.inv(Sigma_hat)) * pos_mean_dev, axis=-1))
    neg_mean_dev = data - Z_hat_neg[:, None]
    exp_neg = np.exp(-0.5 * np.sum(np.matmul(neg_mean_dev, np.linalg.inv(Sigma_hat)) * neg_mean_dev, axis=-1))
    # pos_mask = exp_pos >= exp_neg
    # neg_mask = exp_pos < exp_neg
    q_pos = exp_pos / (exp_pos + exp_neg)
    q_neg = 1 - q_pos
    Z_hat = Z_hat_pos * q_pos + Z_hat_neg * q_neg

    # Mean assignment
    # mean_vec = data.mean(axis=-1)
    # pos_mask = mean_vec >= 0
    # neg_mask = mean_vec < 0
    # mu_bimodal = pos_mask * mu_bar - neg_mask * mu_bar
    # Z_hat = np.matmul(data - mu_bimodal[:, None], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bimodal
    print("Done with {} steps\tExpected error: {}".format(m, v_hat))
    # print("Precision Matrix")
    # print(np.linalg.inv(Sigma_hat))
    weight = np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat).sum(axis=-1) * v_bar
    return Z_hat, weight, Sigma_hat


def EM_bimodal_biased(
    data,
    sigma_bar=2,
    rho_bar=0,
    c=0.1,
    M=10000,
    v_bar=1,
    mu_bar=0,
    assign="mean",
    labels=None,
    m_bar=0,
    tied=True,
):
    N = data.shape[-1]
    Sigma_bar = np.identity(N) * sigma_bar + (np.ones((N, N)) - np.identity(N)) * rho_bar
    Sigma_hat_pos = Sigma_bar
    Sigma_hat_neg = Sigma_bar
    m_hat_pos = - np.ones(N) * m_bar
    m_hat_neg = + np.ones(N) * m_bar
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

        # Compute z_hat based on each of the mode
        Z_hat_pos = np.matmul(data - mu_bar - m_hat_pos[None, :], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat_pos)).sum(axis=-1) * v_bar + mu_bar
        Z_hat_neg = np.matmul(data + mu_bar - m_hat_neg[None, :], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat_neg)).sum(axis=-1) * v_bar - mu_bar

        if assign == "likelihood":
            pos_mean_dev = data - Z_hat_pos[:, None] - m_hat_pos[None, :]
            exp_pos = np.exp(-0.5 * np.sum(np.matmul(pos_mean_dev, np.linalg.inv(Sigma_hat_pos)) * pos_mean_dev, axis=-1))
            neg_mean_dev = data - Z_hat_neg[:, None] - m_hat_neg[None, :]
            exp_neg = np.exp(-0.5 * np.sum(np.matmul(neg_mean_dev, np.linalg.inv(Sigma_hat_neg)) * neg_mean_dev, axis=-1))
            pos_mask = exp_pos >= exp_neg
            neg_mask = exp_pos < exp_neg
        else:
            pos_mask = mean_vec >= 0
            neg_mask = mean_vec < 0
        mu_bimodal = pos_mask * mu_bar - neg_mask * mu_bar
        m_hat = m_hat_pos[None, :] * pos_mask[:, None] + m_hat_neg[None, :] * neg_mask[:, None]

        z_prev = z_hat
        if tied:
            Sigma_hat_inv = np.linalg.inv(Sigma_hat_pos)
            Sigma_hat_pos_inv = Sigma_hat_inv
            Sigma_hat_neg_inv = Sigma_hat_inv
        else:
            Sigma_hat_pos_inv = np.linalg.inv(Sigma_hat_pos)
            Sigma_hat_neg_inv = np.linalg.inv(Sigma_hat_neg)
        z_hat = Z_hat_pos * pos_mask + Z_hat_neg * neg_mask

        if tied:
            v_hat = 1 / (v_bar + Sigma_hat_inv.sum())
            Y_cov = np.matmul((data-z_hat[:, None]-m_hat).transpose(), (data-z_hat[:, None] - m_hat))
            Y_cov += T * v_hat * np.ones((N, N))
            Sigma_hat = Y_cov / T
            Sigma_hat_pos = Sigma_hat
            Sigma_hat_neg = Sigma_hat
        else:
            v_hat_pos = 1 / (1/v_bar + Sigma_hat_pos_inv.sum())
            v_hat_neg = 1 / (1/v_bar + Sigma_hat_neg_inv.sum())
            Y_cov_pos = np.matmul((data-z_hat[:, None]-m_hat).transpose(), (data-z_hat[:, None] - m_hat) * pos_mask[:, None])
            Y_cov_pos += T * v_hat_pos * np.ones((N, N))
            Y_cov_neg = np.matmul((data-z_hat[:, None]-m_hat).transpose(), (data-z_hat[:, None] - m_hat) * neg_mask[:, None])
            Y_cov_neg += T * v_hat_neg * np.ones((N, N))
            Sigma_hat_pos = Y_cov_pos / T
            Sigma_hat_neg = Y_cov_neg / T
        m_hat_pos = (pos_mask[:, None] * (data-z_hat[:, None])).sum(axis=0) / pos_mask.sum()
        m_hat_neg = (neg_mask[:, None] * (data-z_hat[:, None])).sum(axis=0) / neg_mask.sum()
        m_hat = m_hat_pos[None, :] * pos_mask[:, None] + m_hat_neg[None, :] * neg_mask[:, None]
        m += 1
    # Z_hat = np.matmul(data - mu_bimodal[:, None] - m_hat[None, :], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat)).sum(axis=-1) * v_bar + mu_bimodal
    # Sigma_hat_inv = np.linalg.inv(Sigma_hat)
    # v_hat = 1 / (v_bar + Sigma_hat_inv.sum())

    Z_hat_pos = np.matmul(data - mu_bar - m_hat_pos[None, :], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat_pos)).sum(axis=-1) * v_bar + mu_bar
    Z_hat_neg = np.matmul(data + mu_bar - m_hat_neg[None, :], np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat_neg)).sum(axis=-1) * v_bar - mu_bar

    # likelihood assignment
    pos_mean_dev = data - Z_hat_pos[:, None] - m_hat_pos[None, :]
    exp_pos = np.exp(-0.5 * np.sum(np.matmul(pos_mean_dev, np.linalg.inv(Sigma_hat_pos)) * pos_mean_dev, axis=-1))
    neg_mean_dev = data - Z_hat_neg[:, None] - m_hat_neg[None, :]
    exp_neg = np.exp(-0.5 * np.sum(np.matmul(neg_mean_dev, np.linalg.inv(Sigma_hat_neg)) * neg_mean_dev, axis=-1))
    inv_det_Sigma_pos = 1 / np.sqrt(np.linalg.det(Sigma_hat_pos))
    inv_det_Sigma_neg = 1 / np.sqrt(np.linalg.det(Sigma_hat_neg))
    if tied:
        q_pos = exp_pos / (exp_pos + exp_neg)
    else:
        q_pos = (inv_det_Sigma_pos * exp_pos) / (inv_det_Sigma_pos * exp_pos + inv_det_Sigma_neg * exp_neg)
    q_neg = 1 - q_pos
    Z_hat = Z_hat_pos * q_pos + Z_hat_neg * q_neg
    # Z_hat = - np.log(1 / q_pos - 1)

    # Mean assignment
    # mean_vec = data.mean(axis=-1)
    # pos_mask = mean_vec >= 0
    # neg_mask = mean_vec < 0
    # Z_hat = Z_hat_pos * pos_mask + Z_hat_neg * neg_mask

    print("Done with {} steps\tExpected error: {}".format(m, 0))
    # print("Precision Matrix")
    # print(np.linalg.inv(Sigma_hat))
    # weight = np.linalg.inv(np.ones((N, N)) * v_bar + Sigma_hat).sum(axis=-1) * v_bar
    return Z_hat, Sigma_hat_pos, Sigma_hat_neg, m_hat_pos, m_hat_neg


def main(args):
    # model_list = ["llama3", "beluga", "mistral", "zephyr", "starling", "openorca", "dolphin", "mistral1", "hermes2", "hermes25"]
    model_list = ["llama3", "beluga", "mistral", "zephyr", "starling"]
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

        # Direct averaging
        data_tensor = np.minimum(0.9995, data_tensor)
        data = - np.log(1 / data_tensor[:, :, 0] - 1)

    data_mean = data.mean(axis=1)
    pos_center = ((data_mean >= 0) * data_mean).mean()
    neg_center = ((data_mean < 0) * data_mean).mean()
    mu_bar = (pos_center - neg_center) / 2
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
            v_bar=5,
            mu_bar=0,
        )
        pred = 1 / (1 + np.exp(-pred))
        predicts = pred < 0.5
        hits = (labels == predicts).sum()
    elif args.algorithm == "em_bimodal":
        pred, weight, Sigma_hat = EM_bimodal(
            data,
            len(model_list),
            sigma_bar=2,
            rho_bar=0.0,
            c=0,
            M=10000,
            v_bar=v_bar_gen if artificial else 5,
            mu_bar=mu_bar_gen if artificial else 5,
            assign="mean",
            labels=labels,
        )
        print("Actual Estimation of Sigma:")
        print(Sigma_hat)
        print("Actual Estimation of mean:")
        print(pred)
        np.save("outputs/zhat_2_5_gt.npy", pred)
        pred = 1 / (1 + np.exp(-pred))
        predicts = pred < 0.5
        hits = (labels == predicts).sum()
    elif args.algorithm == "em_bimodal_biased":
        pred, Sigma_hat_pos, Sigma_hat_neg, m_hat_pos, m_hat_neg = EM_bimodal_biased(
            data,
            sigma_bar=2,
            rho_bar=0.0,
            c=0,
            M=10000,
            v_bar=v_bar_gen if artificial else 2,
            mu_bar=mu_bar_gen if artificial else 5,
            assign="mean",
            labels=labels,
            m_bar=0,
            tied=True,
        )
        print("Actual Estimation of Bias:")
        print(m_hat_pos, m_hat_neg)
        # np.save("outputs/zhat_2_5_gt.npy", pred)
        print("Actual Estimation of mean:")
        print(pred)
        pred = 1 / (1 + np.exp(-pred))
        predicts = pred < 0.5
        hits = (labels == predicts).sum()
    elif args.algorithm == "em_gmixture":
        pred, Sigma_hat_1, Sigma_hat_2, mu_1, mu_2 = EM_Gmixture(
            data,
            sigma_bar=1,
            rho_bar=0,
            M=100,
            p=0.5,
            mu_1_init=1,
            mu_2_init=-1,
        )
        hits = (labels == pred).sum()
        print("Actual Estimation of Sigma class positive:")
        print(Sigma_hat_1)
        print("Actual Estimation of Sigma class negative:")
        print(Sigma_hat_2)
        print("Mean 1:")
        print(mu_1)
        print("Mean 2:")
        print(mu_2)
        # print("Precision Matrix:")
        # print(np.linalg.inv(Sigma_hat))
    print("EM Acc: {:.3f}".format(hits / len(labels)))
    np.save("outputs/data.npy", data)


if __name__ == "__main__":
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--algorithm",
        type=str,
        default="em_bimodal",
        choices=["em_orig", "em_bimodal", "em_gmixture", "em_bimodal_biased", "em_orig_biased"],
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
