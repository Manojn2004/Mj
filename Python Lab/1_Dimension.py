import sklearn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# # Load Olivetti Face dataset
# faces = fetch_olivetti_faces()
# X = faces.data  # shape: (400, 4096)
# y = faces.target

# # california
# data = fetch_california_housing()
# X = data.data
# y = data.target

# # iris
# data = load_iris()
# X = data.data
# y = data.target

# #cancer
# data = load_breast_cancer()
# X = data.data
# y = data.target

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)

# Plot the 2D projection
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Person ID')
plt.title("Olivetti Faces projected to 2D using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()