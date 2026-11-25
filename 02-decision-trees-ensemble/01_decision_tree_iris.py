"""
Experiment 02 — Part 1: Decision Tree on Iris

Train a DecisionTreeClassifier, evaluate misclassifications,
and plot the tree.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------
# Load Iris
# ---------------------------------------------------------------------------
iris = datasets.load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

print("Iris samples:", X.shape[0], "| features:", X.shape[1])
print("Classes:", list(class_names))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------------------------
# Train decision tree
# ---------------------------------------------------------------------------
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

train_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)

train_misc = int(np.sum(train_preds != y_train))
test_misc = int(np.sum(test_preds != y_test))

print("\nTrain accuracy:", f"{accuracy_score(y_train, train_preds):.4f}")
print("Test accuracy:", f"{accuracy_score(y_test, test_preds):.4f}")
print(f"Train misclassifications: {train_misc}")
print(f"Test misclassifications:  {test_misc}")
print(
    f"Test misclassification %:  {test_misc * 100 / len(y_test):.2f}%"
)
print("\nPredictions:", test_preds)
print("Actual:     ", y_test)
print("\nConfusion matrix (test):\n", confusion_matrix(y_test, test_preds))

# ---------------------------------------------------------------------------
# Plot tree
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 10))
tree.plot_tree(
    clf,
    feature_names=feature_names,
    class_names=list(class_names),
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree — Iris")
plt.tight_layout()
plt.savefig("iris_decision_tree.png", dpi=120)
print("\nSaved plot -> iris_decision_tree.png")
try:
    plt.show()
except Exception:
    pass
