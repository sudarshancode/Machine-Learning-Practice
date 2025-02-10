#Build an Artificial Neural Network by implementing the Back-propagation algorithm
#and test the same using appropriate data set

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate a classification dataset (moon-shaped clusters)
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Predict on test set
y_pred = mlp.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot Decision Boundary
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_transformed = scaler.transform(grid)
Z_pred = mlp.predict(grid_transformed).reshape(xx.shape)

plt.contourf(xx, yy, Z_pred, alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
plt.title("Decision Boundary - MLPClassifier")
plt.show()
