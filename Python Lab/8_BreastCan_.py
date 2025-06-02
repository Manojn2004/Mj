from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Decision tree
dec= DecisionTreeClassifier()
dec.fit(X_train, y_train)
y_pred = dec.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)