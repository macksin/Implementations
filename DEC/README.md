https://arxiv.org/pdf/1511.06335.pdf

## Discussion

* **K-Means**: Distance metrics are limited to the original data space and they tend to be ineffective when input dimensionality is high (Steinbach et al., 2004)
* **Spectral clustering**: Extremely memory expansive

> We take inspiration from parametric t-SNE. Instead of minimizing KL divergence to produce an embedding that is
faithful to distances in the original data space, we define
a centroid-based probability distribution and minimize its
KL divergence to an auxiliary target distribution to simultaneously improve clustering assignment and feature representation. A centroid-based method also has the benefit of
reducing complexity to O(nk), where k is the number of
centroids.

# Deep embedded clustering

- Minimizes the KL divergence to an auxiliary target distribution to simultaneously improve clustering assignment and feature representation
- "In addition, our experiments
show that DEC is significantly less sensitive to the choice
of hyperparameters compared to state-of-the-art methods.
This robustness is an important property of our clustering
algorithm since, when applied to real data, supervision is
not available for hyperparameter cross-validation."
- New metric of optimal clustering based on Mutual Information

