import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate a synthetic nonlinear dataset
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 0.5 * X**2 - X + 2 + np.random.randn(100)  # Quadratic relationship with noise
X = X.reshape(-1, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform features to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit Polynomial Regression
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

# Plot results
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_test, y_test, color='red', label='Test')
plt.scatter(X_test, y_pred, color='green', label='Polynomial Regression Prediction', marker='x')
plt.title('Polynomial Regression (Degree 2) on Synthetic Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Test R^2 Score:", model.score(X_test_poly, y_test))