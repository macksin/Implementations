import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional
from scipy.special import rel_entr, kl_div
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance, MinCovDet


rg = np.random.RandomState(99)

settings = dict(
    max_iter = 1000,
    tol = 1e-4,
    covariance_type = 'full',
    reg_covar = 1e-6,
    n_init = 3,
    # random_state = rg
)

def generate_pdf(
    rs: np.random.RandomState,
    change: Optional[bool] = False
) -> np.array:
    n = 500
    p1 = rs.normal(0, 0.3, n) * 2
    p2 = rs.exponential(0.3, n)
    if change:
        p1 = rs.randint(0, 1000, size=n)/1000
        p2 = p2 + rs.randint(0, 1000, size=n)/1000 
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
X = StandardScaler().fit_transform(X)

aic = []
bic = []
n_components_range = range(1, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, **settings)
    gmm.fit(X)
    bic.append(gmm.bic(X))
    aic.append(gmm.aic(X))

fig, ax1 = plt.subplots()
ax1.plot(n_components_range, bic, label='bic', color='red', marker='x')
plt.legend()
ax2 = ax1.twinx()
ax2.plot(n_components_range, aic, label='aic', color='blue', marker='x')
plt.legend()
plt.show()

N_COMPONENTS=np.argmin(bic) + 1

def get_order(old: GaussianMixture, new: GaussianMixture) -> list:
    nn = NearestNeighbors(n_neighbors=len(old.means_))
    nn.fit(old.means_)
    comparison_ids = []

    kn = nn.kneighbors(
        new.means_,
        return_distance=False,
        n_neighbors=len(new.means_)
    )

    seen = []
    for i, n_points in enumerate(kn):
        for p in n_points:
            if p in seen:
                continue
            else:
                comparison_ids.append((p, i))
                seen.append(p)
                break
    return comparison_ids


def mahalanobis_diff(new_array: np.array, old_array: np.array) -> np.array:
    emp_cov = EmpiricalCovariance(assume_centered=True)
    emp_cov.fit(old_array.reshape(-1, 1))
    distances = emp_cov.mahalanobis(new_array.reshape(-1, 1))
    return distances


def get_entropy(old: GaussianMixture, new: GaussianMixture) -> np.array:
    order = get_order(old, new)
    cov_old, cov_new = old.covariances_, new.covariances_
    cov_old_, cov_new_ = [], []
    for id_old, id_new in order:
        cov_old_.append(cov_old[id_old])
        cov_new_.append(cov_new[id_new])
    # FUNC = lambda x: np.sqrt((np.array(x).ravel() ** 2))
    FUNC = lambda x: MinMaxScaler().fit_transform(np.array(x).ravel().reshape(-1, 1))
    cov_old_ = FUNC(cov_old_)
    cov_new_ = FUNC(cov_new_)
    entropy = rel_entr(cov_new_, cov_old_)
    # entropy = cov_new_ - cov_old_
    return mahalanobis_diff(cov_new_, cov_old_)

# Now, we will have some pdfs
pdfs_range = range(11)
distributions = []
points = [5, 6, 7, 9]
compare = []
gmms = []
for i in pdfs_range:
    gmm = GaussianMixture(n_components=N_COMPONENTS, **settings)
    if i in points:
        print("%d is different" % i)
        X = generate_pdf(rg, change=True)
    else:
        X = generate_pdf(rg)
    gmm.fit(X)
    gmms.append(gmm)

entropy = [sum(np.abs(get_entropy(d2, d1))) for d1, d2 in \
    zip(gmms[:-1], gmms[1:])]

plt.title("KL divergence between COV Matrices of each PDF")
plt.plot(range(len(entropy)), np.log(entropy), marker='x')
for p in points:
    plt.axvline(p)
plt.ylabel("Log_e(Sum(Hs(New, Old)))")
plt.xlabel("Periods")
plt.legend()
plt.show()

plt.title("KL divergence between COV Matrices of each PDF")
plt.plot(range(len(entropy)), entropy, marker='x')
for p in points:
    plt.axvline(p)
plt.ylabel("Sum(Hs(New, Old))")
plt.xlabel("Periods")
plt.legend()
plt.show()
## Testing

def generate_pdf(
    rs: np.random.RandomState,
    change: Optional[bool] = False
) -> np.array:
    n = 1000
    p1 = rs.normal(0, 0.3, n) * 2
    p2 = rs.exponential(0.3, n)
    if change:
        p1 = rs.randint(0, 100, size=n)
        p2 = rs.randint(0, 100, size=n)
    X = np.stack((p1, p2), axis=1)
    return X

X = generate_pdf(rg)
gmm_old = GaussianMixture(n_components=8, **settings)
gmm_old.fit(X)

X = generate_pdf(rg)
gmm_old2 = GaussianMixture(n_components=8, **settings)
gmm_old2.fit(X)

X = generate_pdf(rg, change=True)
gmm_new = GaussianMixture(n_components=8, **settings)
gmm_new.fit(X)

print(get_order(gmm_old, gmm_new))

# Using the Empirical Cov Log Likehood
X_ok_1 = generate_pdf(rg)
X_ok_2 = generate_pdf(rg)
X_nok_1 = generate_pdf(rg, True)
X_nok_2 = generate_pdf(rg, True)
X_ok_3 = generate_pdf(rg)

pdfs = [X_ok_1, X_ok_2, X_nok_1, X_nok_2, X_ok_3]

FUNC = lambda x: StandardScaler().fit_transform(x)
FUNC = lambda x: MinMaxScaler((1e-2, 0.9999)).fit_transform(x)
diff_pdfs = [(FUNC(dnew), FUNC(dold)) for dnew, dold in
    zip(pdfs[1:], pdfs[:-1])]

scores = []
for pdf in diff_pdfs:
    emp_cov = EmpiricalCovariance().fit(pdf[1])
    scores.append(emp_cov.score(pdf[0]))

plt.title("KL divergence between COV Matrices of each PDF")
plt.plot(range(len(scores)), np.log(scores), marker='x')
for p in [1,3]:
    plt.axvline(p)
plt.xlabel("Periods")
plt.legend()
plt.show()