from sklearn.datasets import fetch_olivetti_faces
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Olivetti Face dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces.images  # shape: (400, 64, 64)
X = faces.data  # shape: (400, 4096)
y = faces.target

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.suptitle("First 10 faces from Olivetti dataset")
plt.show()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict and evaluate
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes classification accuracy on Olivetti Faces:", accuracy)