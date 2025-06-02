import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate a synthetic dataset (best suited for LWR demonstration)
# np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X)


# Locally Weighted Regression (LWR) implementation
def lwr_predict(X_train, y_train, x_query, tau=0.5):
    m = X_train.shape[0]
    W = np.exp(-np.square(X_train - x_query) / (2 * tau ** 2))
    W = np.diag(W.flatten())
    X_ = np.hstack((np.ones_like(X_train), X_train))
    x_query_ = np.array([1, x_query])
    theta = np.linalg.pinv(X_.T @ W @ X_) @ (X_.T @ W @ y_train)
    return x_query_ @ theta

# Predict for all test points
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = np.array([lwr_predict(X_train, y_train, x[0], tau=0.5) for x in X_test])

# Plot results
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='red', label='Test')
plt.scatter(X_test, y_pred, color='green', label='LWR Prediction', marker='x')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()