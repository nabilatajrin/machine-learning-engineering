"""
Experiment 07 — SVM Handwritten Digit Recognition

Load digits, visualize a sample, train several SVC variants, report accuracy.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset = load_digits()
print("Dataset data shape:", dataset.data.shape)  # (1797, 64)
print("Dataset images shape:", dataset.images.shape)  # (1797, 8, 8)
print("Total number of images:", len(dataset.images))

# ---------------------------------------------------------------------------
# Visualize one digit
# ---------------------------------------------------------------------------
n = 30
plt.figure(figsize=(4, 4))
plt.gray()
plt.imshow(dataset.images[n], interpolation="nearest")
plt.title(f"Image No. {n} — label {dataset.target[n]}")
plt.axis("off")
plt.tight_layout()
plt.savefig("sample_digit.png", dpi=120)
print("Saved sample_digit.png")
try:
    plt.show()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Features / labels and split
# ---------------------------------------------------------------------------
dataimage_length = len(dataset.images)
X = dataset.images.reshape((dataimage_length, -1))
Y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------------------------------------------------------
# Train multiple SVM models
# ---------------------------------------------------------------------------
model1 = svm.SVC(kernel="linear")
model2 = svm.SVC(kernel="rbf")
model3 = svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.001, C=0.1)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)

print("\nAccuracy of Model 1 (linear):      {:.2f}%".format(accuracy_score(y_test, y_pred1) * 100))
print("Accuracy of Model 2 (rbf):         {:.2f}%".format(accuracy_score(y_test, y_pred2) * 100))
print("Accuracy of Model 3 (gamma=0.001): {:.2f}%".format(accuracy_score(y_test, y_pred3) * 100))
print("Accuracy of Model 4 (gamma=0.001, C=0.1): {:.2f}%".format(accuracy_score(y_test, y_pred4) * 100))

# ---------------------------------------------------------------------------
# Single-sample prediction demo (best of the four: model3 often wins)
# ---------------------------------------------------------------------------
sample_idx = 734
sample = dataset.images[sample_idx].reshape(1, -1)
pred = model3.predict(sample)
print(f"\nSample image index {sample_idx}: predicted={pred[0]}, true={dataset.target[sample_idx]}")

plt.figure(figsize=(3, 3))
plt.imshow(dataset.images[sample_idx], cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Pred: {pred[0]}")
plt.axis("off")
plt.tight_layout()
plt.savefig("prediction_sample.png", dpi=120)
print("Saved prediction_sample.png")
try:
    plt.show()
except Exception:
    pass
