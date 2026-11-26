"""
Experiment 03 — Titanic Survival Prediction using Gaussian Naive Bayes

Columns used (lab style):
  Pclass, Sex (0=female, 1=male), Age, Fare  ->  Survived
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

DATA_PATH = Path(__file__).resolve().parent / "data" / "titanicsurvival.csv"


def load_dataset() -> pd.DataFrame:
    """Prefer local lab CSV; otherwise use seaborn Titanic."""
    if DATA_PATH.is_file():
        print(f"Loading local file: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        # Normalize expected column names if needed
        rename = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl == "survived":
                rename[c] = "Survived"
            elif cl == "pclass":
                rename[c] = "Pclass"
            elif cl == "sex":
                rename[c] = "Sex"
            elif cl == "age":
                rename[c] = "Age"
            elif cl == "fare":
                rename[c] = "Fare"
        df = df.rename(columns=rename)
        return df[["Pclass", "Sex", "Age", "Fare", "Survived"]].copy()

    print("Local CSV not found — loading seaborn 'titanic' dataset")
    import seaborn as sns

    raw = sns.load_dataset("titanic")
    df = pd.DataFrame(
        {
            "Pclass": raw["pclass"],
            "Sex": raw["sex"],
            "Age": raw["age"],
            "Fare": raw["fare"],
            "Survived": raw["survived"],
        }
    )
    return df.dropna(subset=["Survived", "Pclass", "Sex", "Fare"]).copy()


def main() -> None:
    dataset = load_dataset()
    print("Dataset shape:", dataset.shape)
    print("First 5 rows:\n", dataset.head())

    # Encode Sex: female -> 0, male -> 1 (matches lab notebook)
    if dataset["Sex"].dtype == object or str(dataset["Sex"].dtype) == "category":
        dataset["Sex"] = dataset["Sex"].map({"female": 0, "male": 1}).astype(int)

    X = dataset.drop(columns=["Survived"])
    y = dataset["Survived"].astype(int)

    # Fill missing Age with mean
    print("Columns with NA:", list(X.columns[X.isna().any()]))
    if X["Age"].isna().any():
        X["Age"] = X["Age"].fillna(X["Age"].mean())
    print("Columns with NA after fill:", list(X.columns[X.isna().any()]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nSample (pred, actual):")
    print(np.column_stack((y_pred[:15], y_test.to_numpy()[:15])))

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nAccuracy of the Model: {acc:.4f}%")
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
    print("R^2: ", r2_score(y_test, y_pred))

    # Demo prediction (same style as notebook interactive cell)
    example = np.array([[3, 0, 22, 7.25]])  # 3rd class, female, age 22, fare 7.25
    result = model.predict(example)[0]
    print("\nExample person [Pclass=3, Sex=female, Age=22, Fare=7.25]")
    print("Prediction:", result, end=" — ")
    print("might be Survived" if result == 1 else "might not be Survived")


if __name__ == "__main__":
    main()
