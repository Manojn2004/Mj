from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# Test R^2 score
score = knn.score(X_test, y_test)
print("Test R^2 score:", score)
