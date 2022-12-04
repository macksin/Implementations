from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import numpy as np
import time
import os

seed = np.random.RandomState(42)
N_SAMPLES=10_000

X, y = make_moons(n_samples=N_SAMPLES, random_state=seed)
print("y Distribution: ", np.bincount(y))

# Mutual Info
from sklearn.feature_selection import mutual_info_classif

Ymi = mutual_info_classif(y.reshape(-1, 1), y, discrete_features=True)[0]
print("Target Mi ", Ymi)

Xmi = mutual_info_classif(X[:, 0].reshape(-1, 1), y, discrete_features=False)[0]
print("Mutual Info ", Xmi)

# Mutual Different Components
from sklearn.preprocessing import KBinsDiscretizer

mis = {}
for i in range(2, 21):
    est = KBinsDiscretizer(
        n_bins=i, encode="ordinal", strategy="quantile", dtype=np.float32
    )
    Xt = est.fit_transform(X).astype(np.int32)
    m = mutual_info_classif(Xt[:, 0].reshape(-1, 1), y, discrete_features=True)[0]
    mis[i] = m

fig, axs = plt.subplots(figsize=(8, 5))
axs.axhline(Xmi, label="X[:, 0] Mutual Info")
axs.axhline(Ymi, label="Y vs Y Mutual", c='orange')
axs.plot(list(mis.keys()), list(mis.values()), marker='o')
axs.set_xticks(list(mis.keys()))
axs.set_xlabel("Number of quantiles")
axs.set_ylabel("Mutual Information")
axs.set_title("Mutual Info using quantiles in X[:, 0]")
axs.grid()
axs.legend()
plt.show()

est = KBinsDiscretizer(
        n_bins=10, encode="ordinal", strategy="quantile", dtype=np.float32
)
Xt = est.fit_transform(X).astype(np.int32)
print(Xt)
print(np.bincount(Xt[:, 0]))
print(np.bincount(Xt[:, 1]))
print(est.bin_edges_)

fig.savefig('mutual_info_partitioning.png')