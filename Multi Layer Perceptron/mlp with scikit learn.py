import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,    # Total number of samples
    n_features=2,      # Number of features (2 for visualization purposes)
    n_informative=2,   # Number of informative features
    n_redundant=0,     # Number of redundant features
    random_state=42    # Seed for reproducibility
)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # 20% of the data for testing
    random_state=42    # Seed for reproducibility
)

# Step 3: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Create and train the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20, 20), activation='relu', learning_rate_init=0.01, alpha=0.01, max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = mlp.predict(X_test)

# Step 6: Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Step 7: Visualization of decision boundaries
def plot_decision_boundary(X, y, model):
    # Create a mesh grid for the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict on the mesh grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.title("MLP Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(X_test, y_test, mlp)
