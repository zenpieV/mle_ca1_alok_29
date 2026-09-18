#!/usr/bin/env python3
"""
ML CA 1 - Case Study 2: Credit Card Fraud Detection (train / evaluate / submit).

Dataset: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -
284,807 European cardholder transactions from September 2013, of which 492
(0.173%) are frauds. The raw csv exceeds GitHub's 100 MB file limit, so the
repo carries the identical dataset zipped as credit_card_fraud.csv.zip, which
pandas reads transparently.

The pipeline shape is identical to case study 1
(case_study_1_hospital_readmissions.py) with three adaptations for heavily
imbalanced, tree-based learning:

  * XGBoost is the default estimator
  * numeric features are median-imputed but NOT scaled - tree models are
    invariant to monotone rescaling, unlike case study 1's logistic model
  * class imbalance is handled via scale_pos_weight = n_negative / n_positive,
    and average precision (PR-AUC) joins the reported metrics because plain
    accuracy is meaningless at a 0.17% positive rate (predicting "not fraud"
    for every row already scores 99.83%)

Usage
-----
Evaluation only (holdout split of the labelled data)::

    python case_study_2_credit_card_fraud.py --train credit_card_fraud.csv.zip

Train, evaluate, and write submission.csv once a graded test set is available::

    python case_study_2_credit_card_fraud.py --train credit_card_fraud.csv.zip --test <test.csv>
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
    average_precision_score,
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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42

VALID_COMBINATIONS = {
    ("classification", "xgboost"),
    ("classification", "logistic"),
    ("classification", "random_forest"),
    ("regression", "xgboost"),
    ("regression", "linear"),
    ("regression", "random_forest"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", required=True, help="Path to the labelled training csv (or .zip containing one)")
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
        help="Holdout fraction used for evaluation (default: 0.40, as in case study 1)",
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "logistic", "random_forest", "linear"],
        help="Override the estimator (default: xgboost for classification, linear for regression)",
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
    if pd.api.types.is_numeric_dtype(y) and y.nunique() == 2:
        # a two-valued numeric column (e.g. the 0/1 fraud label) is classification
        return "classification"
    return "regression"


def build_pipeline(task: str, model_name: str, X: pd.DataFrame, y: pd.Series | None = None) -> Pipeline:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        # no scaler: tree-based models are invariant to monotone rescaling
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    scale_pos_weight = None
    if task == "classification" and y is not None:
        classes = sorted(pd.unique(y), key=str)
        if len(classes) == 2:
            positive = classes[-1]
            scale_pos_weight = float((y != positive).sum() / (y == positive).sum())

    estimators = {
        ("classification", "xgboost"): lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
        ),
        ("classification", "logistic"): lambda: LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        ("classification", "random_forest"): lambda: RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        ("regression", "xgboost"): lambda: XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="rmse",
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
        classes = list(model.named_steps["model"].classes_)
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
            positive_probabilities = probabilities[:, classes.index(pos_label)]
            metrics["roc_auc"] = roc_auc_score(y_true_binary, positive_probabilities)
            metrics["pr_auc"] = average_precision_score(y_true_binary, positive_probabilities)
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
    model_name = args.model or ("xgboost" if task == "classification" else "linear")
    if (task, model_name) not in VALID_COMBINATIONS:
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

    model = build_pipeline(task, model_name, X, y_train)
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

    final_model = build_pipeline(task, model_name, X, y)
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
