import numpy as np
from EM import get_artificial_data, EM_bimodal, EM_bimodal_biased
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def plot(x, y):
    limits = 30
    nbins = 20
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(15, 5))
    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(x, bins=nbins)
    axs[0].set_xlim(-limits, limits)
    axs[0].set_title("Fake Beluga")
    
    axs[1].hist(y, bins=nbins)
    axs[1].set_xlim(-limits, limits)
    axs[1].set_title("Fake Starling")
    
    # plt.plot(x, y, '.')
    # plt.xlim(-5, 5)
    # plt.ylim(-20, 20)
    plt.savefig("/Users/briansun/Documents/PAPER/Crowdsourcing/plots/histogram_synthetic.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    mu_bar = 5
    sigma_bar = 1
    mean_1 = [5, -5]
    mean_2 = [-4, -6]
    cov_1 = [[0.1, 0.3], [0.3, 20]]
    # cov_2 = [[0.14, 0.14], [-0.14, 20.4]]
    cov_2 = [[0.1, 0.3], [0.3, 50]]
    N = 5000
    Y, labels, Z = get_artificial_data(mu_bar, sigma_bar, mean_1, mean_2, cov_1, cov_2, N)
    x, y = Y.T
    print("Fake Beluga performance: {:.2f}".format(((x > 0) == (Z > 0)).sum()/N*100))
    print("Fake Zephyr performance: {:.2f}".format(((y > 0) == (Z > 0)).sum()/N*100))
    labels = Z > 0

    # plot(x, y)
    # exit(1);

    simple_average = (x + y) / 2
    hits = ((simple_average > 0) == labels).sum()
    print("Performance on simple average: {:.2f}".format(hits / N * 100))

    # z_hat, _, Sigma_hat = EM_bimodal(
    #     Y,
    #     2,
    #     sigma_bar=2,
    #     rho_bar=0,
    #     c=0.1,
    #     M=10000,
    #     v_bar=2,
    #     mu_bar=2,
    #     assign="mean",
    # )
    # hits = ((z_hat > 0) == labels).sum()
    # print("Performance using Bi-modal zero-mean noise: {:.2f}".format(hits / N * 100))

    z_hat, Sigma_hat_pos, Sigma_hat_neg, m_hat_pos, m_hat_neg = EM_bimodal_biased(
        Y,
        sigma_bar=2,
        rho_bar=0,
        c=0.1,
        M=10000,
        v_bar=2, #sigma_bar**2,
        mu_bar=5, #mu_bar,
        assign="likelihood",
        m_bar=0,
        labels=1-labels.astype(float),
        tied=True,
    )
    hits = ((z_hat > 0) == labels).sum()
    print("Predicted positive Sigma: {}".format(Sigma_hat_pos))
    print("Predicted negative Sigma: {}".format(Sigma_hat_neg))
    print("Predicted positive mean: {}".format(m_hat_pos))
    print("True positive mean: {}".format(np.array(mean_2)))
    print("Predicted negative mean: {}".format(m_hat_neg))
    print("True negative mean: {}".format(np.array(mean_1)))
    print("Performance using Bi-modal noise: {:.2f}".format(hits / N * 100))
