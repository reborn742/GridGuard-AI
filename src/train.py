import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/sgcc-extracted/data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def extract_features(df):
    print("Extracting customer consumption features...")

    date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    date_cols.sort(key=lambda x: datetime.strptime(x, "%Y/%m/%d"))

    values = df[date_cols].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    safe = np.where(valid, values, np.nan)

    with np.errstate(all="ignore"):
        mean = np.nanmean(safe, axis=1)
        std = np.nanstd(safe, axis=1)
        median = np.nanmedian(safe, axis=1)
        minimum = np.nanmin(safe, axis=1)
        maximum = np.nanmax(safe, axis=1)
        q25 = np.nanpercentile(safe, 25, axis=1)
        q75 = np.nanpercentile(safe, 75, axis=1)

    features = {
        "valid_readings": count,
        "missing_ratio": 1 - count / values.shape[1],
        "mean_consumption": mean,
        "std_consumption": std,
        "median_consumption": median,
        "min_consumption": minimum,
        "max_consumption": maximum,
        "range_consumption": maximum - minimum,
        "q25_consumption": q25,
        "q75_consumption": q75,
        "iqr_consumption": q75 - q25,
        "zero_ratio": (values == 0).sum(axis=1) / np.maximum(count, 1),
        "coefficient_variation": std / np.maximum(np.abs(mean), 1e-6),
    }

    for i, block in enumerate(np.array_split(values, 8, axis=1), start=1):
        with np.errstate(all="ignore"):
            features[f"period_{i}_mean"] = np.nanmean(block, axis=1)

    features["first_30_day_mean"] = np.nanmean(values[:, :30], axis=1)
    features["last_30_day_mean"] = np.nanmean(values[:, -30:], axis=1)

    X = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
    return X


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }


def main():
    start = time.time()

    if not DATA.exists():
        raise FileNotFoundError(DATA)

    print("Loading dataset...")
    df = pd.read_csv(DATA, low_memory=False)

    y = pd.to_numeric(df["FLAG"], errors="raise").astype(int)
    customer_ids = df["CONS_NO"].astype(str)

    X = extract_features(df)
    print(f"Customers: {len(X):,} | Features: {X.shape[1]}")

    indices = np.arange(len(X))

    # 60% train, 20% validation, 20% final test.
    train_idx, remaining_idx = train_test_split(
        indices, test_size=0.40, random_state=42, stratify=y
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.50,
        random_state=42,
        stratify=y.iloc[remaining_idx]
    )

    X_train_raw = X.iloc[train_idx]
    X_val_raw = X.iloc[val_idx]
    X_test_raw = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    print("Fitting imputer on training data only...")
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X.columns
    ).astype(np.float32)

    X_val = pd.DataFrame(
        imputer.transform(X_val_raw),
        columns=X.columns
    ).astype(np.float32)

    X_test = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X.columns
    ).astype(np.float32)

    print("Training Random Forest...")
    classifier = RandomForestClassifier(
        n_estimators=250,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)

    print("Selecting threshold on validation data...")
    val_probabilities = classifier.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_probabilities
    )

    # F2 gives recall twice the weight of precision.
    f2_scores = (
        5 * precision[:-1] * recall[:-1]
        / (4 * precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f2_scores))
    selected_threshold = float(thresholds[best_index])

    validation_metrics = evaluate(
        y_val, val_probabilities, selected_threshold
    )
    validation_metrics["f2"] = float(f2_scores[best_index])

    print(f"Selected validation threshold: {selected_threshold:.4f}")

    print("Evaluating once on untouched test data...")
    test_probabilities = classifier.predict_proba(X_test)[:, 1]

    test_metrics = evaluate(
        y_test, test_probabilities, selected_threshold
    )

    test_metrics["roc_auc"] = float(
        roc_auc_score(y_test, test_probabilities)
    )
    test_metrics["pr_auc"] = float(
        average_precision_score(y_test, test_probabilities)
    )
    test_metrics["baseline_at_0_50"] = evaluate(
        y_test, test_probabilities, 0.50
    )

    print("Training Isolation Forest on normal training customers...")
    normal_train = X_train.loc[y_train.to_numpy() == 0]

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    anomaly_model.fit(normal_train)

    anomaly_predictions = (
        anomaly_model.predict(X_test) == -1
    ).astype(int)

    test_metrics["anomaly_detection"] = {
        "precision": float(precision_score(
            y_test, anomaly_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, anomaly_predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, anomaly_predictions, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_test, anomaly_predictions, labels=[0, 1]
        ).tolist(),
        "flagged_customers": int(anomaly_predictions.sum()),
    }

    print("Saving artifacts...")
    joblib.dump(classifier, MODEL_DIR / "theft_classifier.joblib")
    joblib.dump(anomaly_model, MODEL_DIR / "anomaly_detector.joblib")
    joblib.dump(imputer, MODEL_DIR / "feature_imputer.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

    metrics = {
        "dataset_customers": int(len(df)),
        "feature_count": int(X.shape[1]),
        "train_customers": int(len(train_idx)),
        "validation_customers": int(len(val_idx)),
        "test_customers": int(len(test_idx)),
        "positive_customers": int(y.sum()),
        "positive_rate": float(y.mean()),
        "selected_threshold": selected_threshold,
        "threshold_selection_metric": "F2",
        "validation": validation_metrics,
        "test": test_metrics,
        "runtime_seconds": round(time.time() - start, 2),
    }

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results = pd.DataFrame({
        "customer_id": customer_ids.iloc[test_idx].to_numpy(),
        "actual_label": y_test.to_numpy(),
        "theft_probability": test_probabilities,
        "predicted_label": (
            test_probabilities >= selected_threshold
        ).astype(int),
        "anomaly_flag": anomaly_predictions,
    })
    results.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    print("\n========== FINAL EVALUATION ==========")
    print(json.dumps(metrics, indent=2))
    print(f"\nRuntime: {metrics['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
