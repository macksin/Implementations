import numpy as np
from timeit import timeit

X = np.random.randint(low=0, high=999, size=1_000_000)
X = X.reshape(-1, 2)
print(X)

method = [
    'inverted_cdf',
    'averaged_inverted_cdf',
    'closest_observation',
    'interpolated_inverted_cdf',
    'hazen',
    'weibull',
    'linear',
    'median_unbiased',
    'normal_unbiased'
]

for m in method:
    t = timeit(f"np.percentile(X, q=50, method='{m}', axis=1)", globals=globals(), number=30)
    print(f"Method = {m} time: {t:4.6} seconds.")

