#!/usr/bin/env python3
"""Train and save sample models for testing and demos.

Creates:
  - sklearn RandomForest classifier trained on the Iris dataset
  - sklearn LinearRegression on a synthetic regression dataset
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent


def train_iris_classifier() -> None:
    """Train and save a RandomForest on the Iris dataset."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Iris classifier accuracy: {accuracy:.4f}")

    path = OUTPUT_DIR / "iris_classifier.joblib"
    joblib.dump(clf, path)
    print(f"Saved to {path}")


def train_regression_model() -> None:
    """Train and save a LinearRegression on synthetic data."""
    X, y = make_regression(n_samples=500, n_features=5, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    r2 = reg.score(X_test, y_test)
    rmse = float(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
    print(f"Regression R²: {r2:.4f}, RMSE: {rmse:.4f}")

    path = OUTPUT_DIR / "regression_model.joblib"
    joblib.dump(reg, path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_iris_classifier()
    train_regression_model()
    print("\nDone — sample models ready.")
