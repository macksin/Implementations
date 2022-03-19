import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from typing import Optional
from scipy.special import kl_div

rg = np.random.RandomState(42)

def generate_pdf(rs: np.random.RandomState, change: Optional[bool] = False) -> np.array:
    p1 = rg.normal(0, 0.5, 100)
    if change:
        p2 = rg.binomial(1.1, 0.7, 100)
    else:
        p2 = rg.exponential(0.1, 100)
    X = np.stack((p1, p2), axis=1)
    return X

X = generate_pdf(rg)

plt.plot(X)
plt.show()

aic = []
bic = []
n_components_range = range(1, 20)
for n_components in n_components_range:

    gmm = GaussianMixture(n_components=n_components)

    gmm.fit(X)

    bic.append(gmm.bic(X))
    aic.append(gmm.aic(X))

plt.plot(n_components_range, bic, label='bic')
plt.plot(n_components_range, aic, label='aic')
plt.legend()
plt.show()

N_COMPONENTS=3
gmm = GaussianMixture(n_components=N_COMPONENTS)

def get_cov(mixture_model: GaussianMixture) -> np.array:
    # I have applied some transformations for the KL part
    # not zero = +1e-16
    # not negative = sqrt(x**2)
    return np.sqrt((mixture_model.covariances_.ravel() + 1e-16) ** 2)

#            x   x
# i =  0 1 2 3 4 5 6 7 8 9
# d1   0 1 2 3 4 5 6 7 8
# d2   1 2 3 4 5 6 7 8 9
# Hs   0 1 2 3 4 5 6 7 8
# OK!

# Now, we will have some pdfs
pdfs_range = range(20)

distributions = []
for i in pdfs_range:
    X = generate_pdf(rg)
    if i == 5 or i == 3:
        X = generate_pdf(rg, change=True)
    gmm.fit(X)
    distributions.append(get_cov(gmm))

# How to transform the Covariance Matrices into
# Probability functions? AKA Removing the negative
# Part?
entropy = [sum(kl_div(d2, d1)) for d1, d2 in zip(distributions[:-1], distributions[1:])]

plt.title("KL divergence between COV Matrices of each PDF")
plt.plot(range(len(entropy)), entropy)
plt.axvline(3, label='expected change')
plt.axvline(5, label='expected change')
plt.legend()
plt.show()