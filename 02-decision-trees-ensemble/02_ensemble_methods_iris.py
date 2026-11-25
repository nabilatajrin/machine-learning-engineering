"""
Experiment 02 — Part 2: Ensemble methods on Iris

Compare:
  - Decision Tree (base)
  - Bagging
  - AdaBoost
  - Random Forest
  - Gradient Boosting
  - SVM (baseline)

Optional GridSearchCV on Random Forest.
"""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

base = DecisionTreeClassifier(random_state=42)

models = {
    "Decision Tree": base,
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42,
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "SVM": SVC(random_state=42),
}


def evaluate(name: str, y_true, y_pred) -> float:
    acc = accuracy_score(y_true, y_pred)
    print(f"{name:20s} Accuracy: {acc:.4f}")
    return acc


print("=" * 50)
print("ENSEMBLE COMPARISON (Iris, test_size=0.3)")
print("=" * 50)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = evaluate(name, y_test, pred)

best_name = max(results, key=results.get)
print(f"\nBest (by test accuracy): {best_name} ({results[best_name]:.4f})")

# ---------------------------------------------------------------------------
# Grid search on Random Forest
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("GRID SEARCH — Random Forest")
print("=" * 50)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)
print("Best params:", grid.best_params_)
evaluate("Tuned Random Forest", y_test, best_pred)
