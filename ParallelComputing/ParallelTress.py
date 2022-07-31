from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import numpy as np
import time
import os
from joblib import Parallel, delayed

seed = np.random.RandomState(42)
N_SAMPLES=10_000
N_ESTIMATORS_LIMIT=1000
CPU_COUNT = os.cpu_count()

X, y = make_moons(n_samples=N_SAMPLES, random_state=seed)

estimators_params = list(range(50, N_ESTIMATORS_LIMIT+1, 50))

start0 = time.time()
for n_estimators in estimators_params:
    rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    start = time.time()
    rfc.fit(X, y)
print("Total time sequential: ", (time.time() - start0))
# 96 seconds

# Parallel computing with some parallelization

def fit(n_estimators):
    rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    start = time.time()
    rfc.fit(X, y)

start0 = time.time()
results = Parallel(n_jobs=-1)(delayed(fit)(i) for i in estimators_params)
print("Total time parallel (loky): ", (time.time() - start0))
# 100 seconds

from joblib import parallel_backend

start0 = time.time()
with parallel_backend('threading', n_jobs=-1):
    results = Parallel()(delayed(fit)(i) for i in estimators_params)
print("Total time parallel (threads): ", (time.time() - start0))