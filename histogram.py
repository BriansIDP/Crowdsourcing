import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# data = np.load("outputs/data.npy")
# gt = np.load("outputs/gt.npy")
data = np.load("outputs/crosscheck.npy")
gt = np.load("outputs/refcheck.npy")
data = - np.log(1 / (data+1e-5) - 1 + 1e-5)
gt = - np.log(1 / (gt+1e-5) - 1 + 1e-5)

limits = 50
nbins = 20

fig, axs = plt.subplots(2, 3, sharey=True, tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0, 0].hist(data[:, 0], bins=nbins)
axs[0, 0].set_xlim(-limits, limits)
axs[0, 0].set_title("Y1")
axs[0, 1].hist(data[:, 1], bins=nbins)
axs[0, 1].set_xlim(-limits, limits)
# axs[0, 1].set_title("Mistral-v0.2")
axs[0, 1].set_title("Y2")
axs[0, 2].hist(data[:, 2], bins=nbins)
axs[0, 2].set_xlim(-limits, limits)
# axs[0, 1].set_title("Mistral-v0.2")
axs[0, 2].set_title("Y3")
axs[1, 0].hist(data[:, 3], bins=nbins)
axs[1, 0].set_xlim(-limits, limits)
# axs[1, 0].set_title("Beluga")
axs[1, 0].set_title("Y4")
# axs[1, 1].hist(data[:, 4], bins=20)
# axs[1, 1].set_xlim(-limits, limits)
# # axs[1, 1].set_title("Zephyr")
# axs[1, 1].set_title("Y5")
# axs[1].hist(dist2, bins=n_bins)
axs[1, 2].hist(gt, bins=nbins)
axs[1, 2].set_xlim(-limits, limits)
# axs[1, 1].set_title("Zephyr")
axs[1, 2].set_title("GT")

# plt.savefig("/Users/briansun/Documents/PAPER/Crowdsourcing/plots/histogram.pdf", format="pdf")
plt.show()
