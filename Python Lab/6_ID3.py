import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the zoo dataset
df = pd.read_csv('zoo.csv')
d=df.info()
head=df.head()
head
# Assume the last column is the class label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert categorical features to numeric
X_encoded = pd.get_dummies(X)

# Fit ID3 (DecisionTreeClassifier with entropy)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_encoded, y)

# Predict on the training data (for demonstration)
y_pred = clf.predict(X_encoded)
accuracy = (y_pred == y).mean()
print("Training accuracy:", accuracy)

# Plot the tree
plt.figure(figsize=(16, 8))
plot_tree(clf, feature_names=X_encoded.columns, class_names=[str(c) for c in set(y)], filled=True)
plt.title("ID3 Decision Tree for zoo.csv")
plt.show()