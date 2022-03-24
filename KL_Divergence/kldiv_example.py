import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional
from scipy.special import rel_entr, kl_div
from sklearn.base import clone

rg = np.random.RandomState(99)

settings = dict(
    max_iter = 1000,
    tol = 1e-4,
    covariance_type = 'full',
    reg_covar = 1e-1,
    n_init = 3,
    random_state = rg
)

def generate_pdf(
    rs: np.random.RandomState,
    change: Optional[bool] = False
) -> np.array:
    n = 1000
    p1 = rs.normal(0, 0.3, n) * 2
    p2 = rs.exponential(0.3, n)
    if change:
        p2 = p2 + rs.randint(0, 100, size=n)
    # p3 = rs.normal(10, 11.5, n)
    # X = np.stack((p1, p2, p3), axis=1)
    X = np.stack((p1, p2), axis=1)
    return X

X_ok = generate_pdf(rg)
X_error = generate_pdf(rg, change=True)
X_t1 = StandardScaler().fit_transform(X_ok)
X_t2 = StandardScaler().fit_transform(X_error)

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].set_title("Distribution")
axs[0].scatter(X_ok[:, 0], X_ok[:, 1])
axs[1].set_title("Different Distribution")
axs[1].scatter(X_error[:, 0], X_error[:, 1])
plt.show()

# ==================================================================

X = generate_pdf(rg)
X_t = StandardScaler().fit_transform(X)

aic = []
bic = []
n_components_range = range(1, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, **settings)
    gmm.fit(X_t)
    bic.append(gmm.bic(X_t))
    aic.append(gmm.aic(X_t))


fig, ax1 = plt.subplots()
ax1.plot(n_components_range, bic, label='bic', color='red', marker='x')
plt.legend()
ax2 = ax1.twinx()
ax2.plot(n_components_range, aic, label='aic', color='blue', marker='x')
plt.legend()
plt.show()


N_COMPONENTS=np.argmin(bic) + 1

def get_cov(mixture_model: GaussianMixture) -> np.array:
    # covs = np.sqrt((mixture_model.covariances_.ravel() + 1e-19) ** 2)
    covs = mixture_model.covariances_.ravel()
    # covs = mixture_model.means_.ravel()
    covs = np.abs(covs)
    # return np.sqrt((covs + 1e-19) ** 2)
    return covs

#            x   x
# i =  0 1 2 3 4 5 6 7 8 9
# d1   0 1 2 3 4 5 6 7 8
# d2   1 2 3 4 5 6 7 8 9
# Hs   0 1 2 3 4 5 6 7 8
# OK!

# Now, we will have some pdfs
pdfs_range = range(11)

distributions = []
points = [5, 6, 7, 9]

compare = []
for i in pdfs_range:
    gmm = GaussianMixture(n_components=N_COMPONENTS, **settings)
    if i in points:
        print("%d is different" % i)
        X = generate_pdf(rg, change=True)
    else:
        X = generate_pdf(rg)
    X_t = StandardScaler().fit_transform(X)
    gmm.fit(X_t)
    if i == 4 or i == 5 or i == 6:
        compare.append(gmm.covariances_.ravel())
    distributions.append(get_cov(gmm))

# =============== print comparison
compare = np.stack(compare, axis=1)
compare[:, 2] = rel_entr(np.sqrt((compare[:, 0] + 1e-19) **2), np.sqrt((compare[:, 1] + 1e-19) **2))
print("Compare: ", compare)

def relative_entropy_sum(d2, d1):
    # d2 = np.sqrt((d2 + 1-19) ** 2)
    # d1 = np.sqrt((d1 + 1-19) ** 2)
    r = rel_entr(d2, d1)
    # r = np.sqrt(r**2)
    # d2 = np.array(d2) + 1e-19
    # d1 = np.array(d1) + 1e-19
    # var = sum(np.sqrt((d2 - d1)**2))
    return sum(r)

def relative_entropy_alternative(d2, d1):
    entropy = []
    for i in range(len(d1)):
        _d1, _d2 = abs(d1[i]), abs(d2[i])
        less = min([_d1, _d2])
        more = max([_d1, _d2])
        entropy.append(rel_entr(more, less))
    entropy = np.array(entropy)
    entropy = np.sqrt(entropy ** 2)
    return sum(entropy)



entropy = [relative_entropy_sum(d2, d1) for d1, d2 in \
    zip(distributions[:-1], distributions[1:])]

plt.title("KL divergence between COV Matrices of each PDF")
plt.plot(range(len(entropy)), entropy, marker='x')
for p in points:
    plt.axvline(p)
plt.legend()
plt.show()
