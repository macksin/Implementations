from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from kpca import rbf_kernel_pca

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y == 0, 0], X[y == 0, 1], 
            color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], 
            color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()

X_kpca = rbf_kernel_pca(X, gamma=20, n_components=2)
plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], 
            color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], 
            color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()