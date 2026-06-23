from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.common.io import write_records


def load_ctg_splits(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    frame = pd.read_csv(data_path)
    required_columns = {"binary_target", "split"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required CTG columns: {sorted(missing)}")

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"case_id", "nsp_state", "binary_target", "binary_state", "split"}
    ]
    train_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    test_frame = frame.loc[frame["split"] == "test"].copy()
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Expected non-empty train, validation, and test splits in processed CTG data")
    return train_frame, validation_frame, test_frame, feature_columns


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.1, 0.9, num=17)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, predictions)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _sanitize_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame.loc[:, feature_columns].astype(float)
    sanitized = np.nan_to_num(features.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(sanitized, columns=feature_columns, index=frame.index)


def _evaluate_classifier(
    name: str,
    model,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    train_x = _sanitize_features(train_frame, feature_columns)
    train_y = train_frame["binary_target"].to_numpy(dtype=int)
    validation_x = _sanitize_features(validation_frame, feature_columns)
    validation_y = validation_frame["binary_target"].to_numpy(dtype=int)
    test_x = _sanitize_features(test_frame, feature_columns)
    test_y = test_frame["binary_target"].to_numpy(dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(train_x, train_y)
        validation_probabilities = model.predict_proba(validation_x)[:, 1]
        test_probabilities = model.predict_proba(test_x)[:, 1]
    threshold = _select_threshold(validation_y, validation_probabilities)
    test_predictions = (test_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(test_y, test_predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy_score(test_y, test_predictions), 6),
        "balanced_accuracy": round(balanced_accuracy_score(test_y, test_predictions), 6),
        "roc_auc": round(roc_auc_score(test_y, test_probabilities), 6),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "test_size": int(len(test_frame)),
    }


def run_biomedical_baselines(
    data_path: str | Path,
    output_path: str | Path,
    seed: int = 7,
) -> Path:
    train_frame, validation_frame, test_frame, feature_columns = load_ctg_splits(data_path)

    models = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
                solver="liblinear",
            ),
        ),
        (
            "calibrated_logistic_regression",
            CalibratedClassifierCV(
                estimator=LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="liblinear",
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "calibrated_random_forest",
            CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                min_child_weight=1.0,
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
            ),
        ),
        (
            "calibrated_xgboost",
            CalibratedClassifierCV(
                estimator=XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    random_state=seed,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
        (
            "lightgbm",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        (
            "calibrated_lightgbm",
            CalibratedClassifierCV(
                estimator=LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ]
    records = [
        _evaluate_classifier(
            name=name,
            model=model,
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
            feature_columns=feature_columns,
        )
        for name, model in models
    ]
    return write_records(records, output_path)
