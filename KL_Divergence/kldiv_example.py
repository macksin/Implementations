import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional
from scipy.special import kl_div

rg = np.random.RandomState(42)

def generate_pdf(
    rs: np.random.RandomState,
    change: Optional[bool] = False
) -> np.array:
    n = 150
    p1 = rs.normal(0, 0.5, n)
    if change:
        p2 = rs.binomial(1.1, 0.7, n)
    else:
        p2 = rs.exponential(0.1, n)
    p3 = rs.normal(10, 11.5, n)
    X = np.stack((p1, p2, p3), axis=1)
    return X

X = generate_pdf(rg)
X_t = StandardScaler().fit_transform(X)

plt.scatter(X_t[:, 0], X_t[:, 1])
plt.show()

aic = []
bic = []
n_components_range = range(1, 10)
for n_components in n_components_range:

    gmm = GaussianMixture(n_components=n_components)

    X_t = StandardScaler().fit_transform(X)
    gmm.fit(X_t)

    bic.append(gmm.bic(X))
    aic.append(gmm.aic(X))

fig, ax1 = plt.subplots()
ax1.plot(n_components_range, bic, label='bic', color='red', marker='x')
plt.legend()
ax2 = ax1.twinx()
ax2.plot(n_components_range, aic, label='aic', color='blue', marker='x')
plt.legend()
plt.show()


N_COMPONENTS=2
gmm = GaussianMixture(n_components=N_COMPONENTS)

def get_cov(mixture_model: GaussianMixture) -> np.array:
    # I have applied some transformations for the KL part
    # not zero = +1e-16
    # not negative = sqrt(x**2)
    # covs = np.array([np.cov(x) for x in mixture_model.covariances_])
    # return np.sqrt((covs.ravel() + 1e-16) ** 2)
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
    X_t = StandardScaler().fit_transform(X)
    gmm.fit(X_t)
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
