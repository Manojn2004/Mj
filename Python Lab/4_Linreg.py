import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
# Generate a synthetic linear dataset ,boston california

# X = np.linspace(0, 10, 100)
# y = 2 * X + 1 + np.random.randn(100)  # y = 2x + 1 + noise
# X = X.reshape(-1, 1)


# california
data = fetch_california_housing()
X = data.data
y = data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Fit Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Plot results
# plt.scatter(X_train, y_train, color='blue', label='Train')
# plt.scatter(X_test, y_test, color='red', label='Test')
plt.plot(X_test, y_pred, color='green', label='Linear Regression Prediction')
plt.title('Linear Regression on Synthetic Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print("Model Coefficient (slope):", lr.coef_[0])
print("Model Intercept:", lr.intercept_)
print("Test R^2 Score:", lr.score(X_test, y_test))