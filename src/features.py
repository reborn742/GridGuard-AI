import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "models" / "consumption_schema.json"


def load_schema(path=SCHEMA_PATH):
    with open(path, encoding="utf-8") as file:
        schema = json.load(file)

    if not schema.get("date_columns"):
        raise ValueError("Schema contains no consumption date columns.")

    if schema.get("reading_count") != len(schema["date_columns"]):
        raise ValueError("Schema reading count does not match its date columns.")

    return schema


def extract_features(df, schema):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input data must be a non-empty pandas DataFrame.")

    id_column = schema["customer_id_column"]
    date_columns = schema["date_columns"]

    if df.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names.")

    if id_column not in df.columns:
        raise ValueError(f"Missing customer ID column: {id_column}")

    missing = [column for column in date_columns if column not in df.columns]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Input is missing {len(missing)} required consumption columns: "
            f"{preview}"
        )

    ids = df[id_column]

    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Customer IDs cannot be empty.")

    if ids.astype(str).duplicated().any():
        raise ValueError("Customer IDs must be unique within the input file.")

    values = df[date_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32, copy=True)

    # Infinite values are treated as missing readings.
    values[~np.isfinite(values)] = np.nan

    valid = np.isfinite(values)
    count = valid.sum(axis=1)

    if (count == 0).any():
        rows = np.flatnonzero(count == 0).tolist()
        raise ValueError(
            f"Customers have no valid consumption readings. Row indexes: {rows[:10]}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        mean = np.nanmean(values, axis=1)
        std = np.nanstd(values, axis=1)
        median = np.nanmedian(values, axis=1)
        minimum = np.nanmin(values, axis=1)
        maximum = np.nanmax(values, axis=1)
        q25 = np.nanpercentile(values, 25, axis=1)
        q75 = np.nanpercentile(values, 75, axis=1)

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

        for i, block in enumerate(
            np.array_split(values, 8, axis=1), start=1
        ):
            features[f"period_{i}_mean"] = np.nanmean(block, axis=1)

        features["first_30_day_mean"] = np.nanmean(values[:, :30], axis=1)
        features["last_30_day_mean"] = np.nanmean(values[:, -30:], axis=1)

    X = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)

    return ids.astype(str).reset_index(drop=True), X
