import numpy as np
import matplotlib.pyplot as plt

# 1. Synthetic linear dataset (for regression)
np.random.seed(0)
X_linear = np.linspace(0, 10, 100)
y_linear = 2 * X_linear + 1 + np.random.randn(100)  # y = 2x + 1 + noise

plt.figure()
plt.scatter(X_linear, y_linear, color='blue')
plt.title("Synthetic Linear Dataset")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# 2. Synthetic nonlinear dataset (for polynomial regression)
X_poly = np.linspace(0, 10, 100)
y_poly = 0.5 * X_poly**2 - X_poly + 2 + np.random.randn(100)  # Quadratic with noise

plt.figure()
plt.scatter(X_poly, y_poly, color='green')
plt.title("Synthetic Nonlinear (Quadratic) Dataset")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# 3. Synthetic classification dataset (two classes)
from sklearn.datasets import make_classification

X_class, y_class = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                       n_clusters_per_class=1, n_classes=2, random_state=0)

plt.figure()
plt.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='bwr')
plt.title("Synthetic Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()