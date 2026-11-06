import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import ROOT, extract_features, load_schema


MODEL_DIR = ROOT / "models"
MAX_INPUT_FILE_BYTES = 25 * 1024 * 1024
MAX_CUSTOMERS = 10_000


def predict_customers(input_path, output_path, threshold=None):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if input_path.stat().st_size > MAX_INPUT_FILE_BYTES:
        raise ValueError("Input CSV exceeds the 25 MB file-size limit.")

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)

        if not header:
            raise ValueError("Input CSV is empty or has no header.")

        customer_count = sum(1 for _ in reader)

    if customer_count == 0:
        raise ValueError("Input CSV contains no customer records.")

    if customer_count > MAX_CUSTOMERS:
        raise ValueError(
            f"Input contains {customer_count:,} customers. "
            f"The maximum allowed is {MAX_CUSTOMERS:,}."
        )

    schema = load_schema()

    classifier = joblib.load(MODEL_DIR / "theft_classifier.joblib")
    imputer = joblib.load(MODEL_DIR / "feature_imputer.joblib")
    anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.joblib")
    expected_features = joblib.load(MODEL_DIR / "feature_columns.joblib")

    if classifier.n_features_in_ != len(expected_features):
        raise ValueError("Classifier feature count does not match saved feature list.")

    if imputer.n_features_in_ != len(expected_features):
        raise ValueError("Imputer feature count does not match saved feature list.")

    if len(expected_features) != schema["feature_count"]:
        raise ValueError("Schema feature count does not match trained model.")

    metrics_path = ROOT / "reports" / "metrics.json"
    import json

    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    selected_threshold = float(
        metrics["selected_threshold"] if threshold is None else threshold
    )

    if not 0 < selected_threshold < 1:
        raise ValueError("Threshold must be greater than 0 and less than 1.")

    df = pd.read_csv(input_path, low_memory=False)

    customer_ids, features = extract_features(df, schema)

    if list(features.columns) != list(expected_features):
        raise ValueError("Extracted feature order differs from the trained model.")

    transformed = pd.DataFrame(
        imputer.transform(features),
        columns=expected_features
    ).astype(np.float32)

    probabilities = classifier.predict_proba(transformed)[:, 1]

    anomaly_predictions = anomaly_model.predict(transformed)
    anomaly_scores = -anomaly_model.decision_function(transformed)

    if not np.isfinite(probabilities).all():
        raise ValueError("Classifier returned non-finite probabilities.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier returned probabilities outside [0, 1].")

    if not np.isin(anomaly_predictions, [-1, 1]).all():
        raise ValueError("Anomaly detector returned invalid predictions.")

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("Anomaly detector returned non-finite scores.")

    anomaly_flags = (anomaly_predictions == -1).astype(int)

    predicted = (probabilities >= selected_threshold).astype(int)

    risk_levels = np.select(
        [
            probabilities >= 0.75,
            probabilities >= selected_threshold
        ],
        [
            "High",
            "Elevated"
        ],
        default="Low"
    )

    results = pd.DataFrame({
        "customer_id": customer_ids,
        "theft_probability": probabilities,
        "predicted_label": predicted,
        "risk_level": risk_levels,
        "anomaly_flag": anomaly_flags,
        "anomaly_score": anomaly_scores
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\n========== INFERENCE SUMMARY ==========")
    print(f"Input customers: {len(results):,}")
    print(f"Risk threshold: {selected_threshold:.4f}")
    print(f"Flagged customers: {int(predicted.sum()):,}")
    print(f"High risk: {(risk_levels == 'High').sum():,}")
    print(f"Elevated risk: {(risk_levels == 'Elevated').sum():,}")
    print(f"Anomaly flags: {int(anomaly_flags.sum()):,}")
    print(f"Output: {output_path.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score customer electricity consumption records."
    )
    parser.add_argument("input", help="Path to input customer CSV.")
    parser.add_argument(
        "--output",
        default="reports/customer_predictions.csv",
        help="Output prediction CSV path."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional risk threshold. Defaults to validation-selected threshold."
    )

    args = parser.parse_args()

    predict_customers(
        args.input,
        args.output,
        args.threshold
    )


if __name__ == "__main__":
    main()
