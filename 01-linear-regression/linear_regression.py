"""
Experiment 01 — Linear Regression

1) Manual slope & intercept from sum formulas
2) sklearn LinearRegression + train/test split
3) Metrics and scatter + line plot
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Data (independent x, dependent y)
# ---------------------------------------------------------------------------
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12], dtype=float)

n = len(x)
sumx = np.sum(x)
sumy = np.sum(y)
sum_squared = np.sum(x**2)
sum_XY = np.sum(x * y)

print("=" * 60)
print("MANUAL LINEAR REGRESSION")
print("=" * 60)
print(f"n            = {n}")
print(f"sum(x)       = {sumx}")
print(f"sum(y)       = {sumy}")
print(f"sum(x^2)     = {sum_squared}")
print(f"sum(x*y)     = {sum_XY}")

# slope m and intercept c
slope = (n * sum_XY - sumx * sumy) / (n * sum_squared - sumx**2)
intercept = (sumy - slope * sumx) / n
print(f"\nManual slope (m)      = {slope:.6f}")
print(f"Manual intercept (c)  = {intercept:.6f}")
print(f"Equation: y = {slope:.4f} * x + {intercept:.4f}")

# ---------------------------------------------------------------------------
# sklearn model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SKLEARN LINEAR REGRESSION")
print("=" * 60)

x_train, x_test, y_train, y_test = train_test_split(
    x.reshape(-1, 1),
    y,
    test_size=0.2,
    random_state=38,
)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

model_slope = float(regressor.coef_[0])
model_intercept = float(regressor.intercept_)
print(f"sklearn slope     = {model_slope:.6f}")
print(f"sklearn intercept = {model_intercept:.6f}")

y_pred = regressor.predict(x_test)
print("\nTest X:", x_test.ravel())
print("Test y true:", y_test)
print("Test y pred:", np.round(y_pred, 3))

# Sample predictions
for val in (3, 20):
    p = regressor.predict(np.array([[val]]))[0]
    print(f"predict(x={val}) -> {p:.3f}")

print("\nMetrics on test set:")
print(f"  R2   = {r2_score(y_test, y_pred):.4f}")
print(f"  MSE  = {mean_squared_error(y_test, y_pred):.4f}")
print(f"  MAE  = {mean_absolute_error(y_test, y_pred):.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# Smooth line over full x range for a clearer figure
x_line = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
y_line = regressor.predict(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="blue", label="Training data")
plt.scatter(x_test, y_test, color="green", label="Test data")
plt.plot(x_line, y_line, color="red", label="Regression line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("linear_regression_plot.png", dpi=120)
print("\nSaved plot -> linear_regression_plot.png")
try:
    plt.show()
except Exception:
    pass
