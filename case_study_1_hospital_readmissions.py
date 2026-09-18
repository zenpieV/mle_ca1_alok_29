#!/usr/bin/env python3
"""
ML CA 1 - train / evaluate / submit pipeline.

A corrected, reusable version of the instructor's example notebook. The example
contained several bugs, fixed here:

  * ``X.select_dtypes(...).column`` -> ``.columns``
  * the train/validation split unpacked tuples without calling ``train_test_split``
  * estimators were passed as classes (``LinearRegression``) instead of instances
  * categorical string columns were imputed but never encoded, which crashes
    linear models - they are now one-hot encoded with ``handle_unknown="ignore"``
    so unseen categories in the test set do not break prediction
  * a yes/no target is a classification problem, so it trains LogisticRegression
    and reports accuracy / precision / recall / F1 / ROC-AUC instead of the
    regression metrics (RMSE / MAE / R2 are used for numeric targets)

Works for any dataset: pass --target / --id-column to override auto-detection.

Usage
-----
Evaluation only (holdout split of the labelled data)::

    python main.py --train hospital_readmissions.csv

Train, evaluate, and write submission.csv once the graded test set is available::

    python main.py --train hospital_readmissions.csv --test hospital_readmissions_test.csv
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", required=True, help="Path to the labelled training csv")
    parser.add_argument(
        "--test",
        help="Path to the unlabelled test csv; providing it enables submission.csv output",
    )
    parser.add_argument(
        "--target",
        help="Target column name (default: the last column of the train csv)",
    )
    parser.add_argument(
        "--id-column",
        help="ID column carried into submission.csv (default: auto-detect a column "
        "named 'id' or ending in '_id'; otherwise rows are keyed by index)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.40,
        help="Holdout fraction used for evaluation (default: 0.40, as in the example notebook)",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "logistic", "random_forest"],
        help="Override the estimator (defaults: logistic for classification, linear for regression)",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output path for the submission file (default: submission.csv)",
    )
    return parser.parse_args()


def detect_task(y: pd.Series) -> str:
    if (
        pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
    ):
        return "classification"
    return "regression"


def build_pipeline(task: str, model_name: str, X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    estimators = {
        ("classification", "logistic"): lambda: LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        ("classification", "random_forest"): lambda: RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        ("regression", "linear"): LinearRegression,
        ("regression", "random_forest"): lambda: RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimators[(task, model_name)]()),
    ])


def evaluate(task: str, model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    predictions = model.predict(X_valid)

    if task == "classification":
        estimator = model.named_steps["model"]
        classes = list(estimator.classes_)
        if len(classes) == 2:
            average, pos_label = "binary", classes[-1]
        else:
            average, pos_label = "weighted", None
        metrics = {
            "accuracy": accuracy_score(y_valid, predictions),
            "precision": precision_score(
                y_valid, predictions, average=average, pos_label=pos_label, zero_division=0
            ),
            "recall": recall_score(
                y_valid, predictions, average=average, pos_label=pos_label, zero_division=0
            ),
            "f1": f1_score(
                y_valid, predictions, average=average, pos_label=pos_label, zero_division=0
            ),
        }
        probabilities = model.predict_proba(X_valid)
        if len(classes) == 2:
            y_true_binary = (pd.Series(y_valid) == pos_label).astype(int)
            metrics["roc_auc"] = roc_auc_score(
                y_true_binary, probabilities[:, classes.index(pos_label)]
            )
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_valid, probabilities, multi_class="ovr", average="weighted", labels=classes
            )
    else:
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_valid, predictions))),
            "mae": mean_absolute_error(y_valid, predictions),
            "r2": r2_score(y_valid, predictions),
        }
    return metrics


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.train)
    target = args.target or train.columns[-1]
    if target not in train.columns:
        sys.exit(
            f"Target column '{target}' not found. Available columns: {list(train.columns)}"
        )

    id_column = args.id_column
    if id_column is None:
        id_candidates = [
            c
            for c in train.columns
            if c != target and re.match(r"(?:^|_)id$", c, flags=re.IGNORECASE)
        ]
        id_column = id_candidates[0] if id_candidates else None

    drop_columns = [target] + ([id_column] if id_column in train.columns else [])
    X = train.drop(columns=drop_columns)
    y = train[target]

    task = detect_task(y)
    model_name = args.model or ("logistic" if task == "classification" else "linear")
    if (task, model_name) not in {
        ("classification", "logistic"),
        ("classification", "random_forest"),
        ("regression", "linear"),
        ("regression", "random_forest"),
    }:
        sys.exit(f"Model '{model_name}' cannot be used for a {task} target.")

    print(f"Train data : {train.shape[0]} rows x {train.shape[1]} columns ({args.train})")
    print(f"Target     : '{target}' ({task})")
    print(f"ID column  : {id_column if id_column else 'none found - submissions will use row index'}")
    print(f"Model      : {model_name}")
    if task == "classification":
        print(f"Classes    : {y.value_counts().to_dict()}")
    print()

    stratify = y if task == "classification" and y.value_counts().min() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    model = build_pipeline(task, model_name, X)
    model.fit(X_train, y_train)
    metrics = evaluate(task, model, X_valid, y_valid)

    print(f"Evaluation on {args.test_size:.0%} holdout ({len(X_valid)} rows):")
    for name, value in metrics.items():
        print(f"  {name:<9} {value:.4f}")

    if not args.test:
        print(
            "\nNo --test csv provided, so no submission.csv was written. "
            "Once the graded test set is available, rerun with: "
            f"--test <test.csv> to produce {args.out}"
        )
        return

    test = pd.read_csv(args.test)
    if target in test.columns:
        print(f"\nNote: test csv already contains '{target}' - dropping it before predicting.")
        test = test.drop(columns=[target])

    feature_columns = [c for c in test.columns if c != id_column]
    X_test = test[feature_columns]

    final_model = build_pipeline(task, model_name, X)
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test)

    if id_column and id_column in test.columns:
        submission = pd.DataFrame({id_column: test[id_column], target: test_predictions})
    else:
        submission = pd.DataFrame({"row_id": test.index, target: test_predictions})

    submission.to_csv(args.out, index=False)
    print(f"\nFinal model refit on all {len(X)} labelled rows.")
    print(f"Wrote {args.out} ({len(submission)} predictions, columns: {list(submission.columns)})")


if __name__ == "__main__":
    main()
