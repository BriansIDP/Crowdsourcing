import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter


data = np.load("outputs/data.npy")
gt = np.load("outputs/gt.npy")
# data = np.load("outputs/crosscheck.npy")
# gt = np.load("outputs/refcheck.npy")
# data = - np.log(1 / (data+1e-5) - 1 + 1e-5)
# gt = - np.log(1 / (gt+1e-5) - 1 + 1e-5)

limits = 20
nbins = 20

data = np.load("outputs/zhat_2_1.npy")
fig, axs = plt.subplots(2, 3, sharey=True, tight_layout=True, figsize=(12, 6))
# We can set the number of bins with the *bins* keyword argument.
axs[0, 0].hist(data, bins=nbins)
axs[0, 0].set_xlim(-limits, limits)
# axs[0, 0].set_title("Llama-3")
axs[0, 0].set_title("Mean=1.0, Var=2.0, Acc=74.0")

data = np.load("outputs/zhat_2_2.npy")
axs[0, 1].hist(data, bins=nbins)
axs[0, 1].set_xlim(-limits, limits)
# axs[0, 1].set_title("Mistral-v0.2")
# axs[0, 1].set_title("Beluga")
axs[0, 1].set_title("Mean=2.0, Var=2.0, Acc=75.8")

data = np.load("outputs/zhat_2_5.npy")
axs[0, 2].hist(data, bins=nbins)
axs[0, 2].set_xlim(-limits, limits)
# axs[0, 1].set_title("Mistral-v0.2")
# axs[0, 2].set_title("Mistral")
axs[0, 2].set_title("Mean=5.0, Var=2.0, Acc=74.9")

data = np.load("outputs/zhat_2_1_gt.npy")
axs[1, 0].hist(data, bins=nbins)
axs[1, 0].set_xlim(-limits, limits)
# axs[1, 0].set_title("Beluga")
# axs[1, 0].set_title("Zephyr")
axs[1, 0].set_title("Mean=1.0, Var=2.0, Acc=72.7, Groundtruth")

data = np.load("outputs/zhat_2_2_gt.npy")
axs[1, 1].hist(data, bins=nbins)
axs[1, 1].set_xlim(-limits, limits)
# axs[1, 1].set_title("Zephyr")
axs[1, 1].set_title("Starling")
# axs[1].hist(dist2, bins=n_bins)
axs[1, 1].set_title("Mean=2.0, Var=2.0, Acc=76.2, Groundtruth")

data = np.load("outputs/zhat_2_5_gt.npy")
axs[1, 2].hist(data, bins=nbins)
axs[1, 2].set_xlim(-limits, limits)
# axs[1, 1].set_title("Zephyr")
# axs[1, 2].set_title("OpenOrca")
axs[1, 2].set_title("Mean=5.0, Var=2.0, Acc=74.9, Groundtruth")

# axs[2, 0].hist(data[:, 6], bins=nbins)
# axs[2, 0].set_xlim(-limits, limits)
# # axs[1, 1].set_title("Zephyr")
# axs[2, 0].set_title("GPT-3 (SelfCheck)")
# 
# axs[2, 1].hist(gt, bins=nbins)
# axs[2, 1].set_xlim(-limits, limits)
# # axs[1, 1].set_title("Zephyr")
# axs[2, 1].set_title("Annotation")

plt.savefig("/Users/briansun/Documents/PAPER/Crowdsourcing/plots/histogram_haludial_pred.pdf", format="pdf")
plt.show()
