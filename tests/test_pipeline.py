import tempfile
from contextlib import ExitStack
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import ROOT, extract_features, load_schema
from src.inference import predict_customers


class GridGuardPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema()
        cls.date_columns = cls.schema["date_columns"]
        cls.expected_features = joblib.load(
            ROOT / "models" / "feature_columns.joblib"
        )

    def make_input(self, rows=2):
        reading_count = len(self.date_columns)

        base = np.linspace(0.1, 2.0, reading_count)

        data = {
            "CONS_NO": [
                f"TEST_CUSTOMER_{i + 1:03d}"
                for i in range(rows)
            ]
        }

        for index, column in enumerate(self.date_columns):
            data[column] = [
                float(base[index] * (1 + customer * 0.15))
                for customer in range(rows)
            ]

        return pd.DataFrame(data)

    def run_inference_with_mocked_outputs(
        self, temp_dir, probabilities=None, anomaly_predictions=None, anomaly_scores=None
    ):
        df = self.make_input(rows=2)
        input_path = Path(temp_dir) / "input.csv"
        output_path = Path(temp_dir) / "predictions.csv"
        df.to_csv(input_path, index=False)

        classifier = joblib.load(ROOT / "models" / "theft_classifier.joblib")
        imputer = joblib.load(ROOT / "models" / "feature_imputer.joblib")
        anomaly_model = joblib.load(ROOT / "models" / "anomaly_detector.joblib")

        with ExitStack() as stack:
            stack.enter_context(patch(
                "src.inference.joblib.load",
                side_effect=[classifier, imputer, anomaly_model, self.expected_features]
            ))

            if probabilities is not None:
                stack.enter_context(patch.object(
                    classifier, "predict_proba", return_value=probabilities
                ))

            if anomaly_predictions is not None:
                stack.enter_context(patch.object(
                    anomaly_model, "predict", return_value=anomaly_predictions
                ))

            if anomaly_scores is not None:
                stack.enter_context(patch.object(
                    anomaly_model, "decision_function",
                    return_value=-np.asarray(anomaly_scores)
                ))

            return predict_customers(input_path, output_path)

    def test_schema_matches_saved_model(self):
        self.assertEqual(
            self.schema["reading_count"],
            len(self.date_columns)
        )
        self.assertEqual(
            self.schema["feature_count"],
            len(self.expected_features)
        )
        self.assertEqual(len(self.expected_features), 23)

    def test_feature_extraction_order_and_shape(self):
        df = self.make_input(rows=3)

        customer_ids, features = extract_features(df, self.schema)

        self.assertEqual(len(customer_ids), 3)
        self.assertEqual(features.shape, (3, 23))
        self.assertEqual(
            list(features.columns),
            list(self.expected_features)
        )

    def test_non_numeric_reading_is_treated_as_missing(self):
        df = self.make_input(rows=1)
        df[self.date_columns[0]] = df[self.date_columns[0]].astype(object)
        df.loc[0, self.date_columns[0]] = "invalid"

        _, features = extract_features(df, self.schema)

        self.assertEqual(
            int(features.loc[0, "valid_readings"]),
            len(self.date_columns) - 1
        )

    def test_missing_consumption_column_is_rejected(self):
        df = self.make_input(rows=1)
        df = df.drop(columns=[self.date_columns[-1]])

        with self.assertRaisesRegex(ValueError, "missing"):
            extract_features(df, self.schema)

    def test_duplicate_customer_ids_are_rejected(self):
        df = self.make_input(rows=2)
        df.loc[1, "CONS_NO"] = df.loc[0, "CONS_NO"]

        with self.assertRaisesRegex(ValueError, "unique"):
            extract_features(df, self.schema)

    def test_empty_customer_id_is_rejected(self):
        df = self.make_input(rows=1)
        df.loc[0, "CONS_NO"] = ""

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            extract_features(df, self.schema)

    def test_customer_with_no_valid_readings_is_rejected(self):
        df = self.make_input(rows=1)

        for column in self.date_columns:
            df[column] = df[column].astype(object)
            df.loc[0, column] = "invalid"

        with self.assertRaisesRegex(ValueError, "no valid"):
            extract_features(df, self.schema)

    def test_duplicate_input_columns_are_rejected(self):
        df = self.make_input(rows=1)
        df = pd.concat(
            [df, df[[self.date_columns[0]]]],
            axis=1
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            extract_features(df, self.schema)

    def test_end_to_end_customer_inference(self):
        df = self.make_input(rows=3)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            output_path = Path(temp_dir) / "predictions.csv"

            df.to_csv(input_path, index=False)

            results = predict_customers(
                input_path,
                output_path
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(len(results), 3)

            expected_columns = [
                "customer_id",
                "theft_probability",
                "predicted_label",
                "risk_level",
                "anomaly_flag",
                "anomaly_score"
            ]

            self.assertEqual(
                list(results.columns),
                expected_columns
            )

            self.assertTrue(
                results["theft_probability"].between(0, 1).all()
            )

            self.assertTrue(
                results["predicted_label"].isin([0, 1]).all()
            )

            self.assertTrue(
                results["risk_level"].isin(
                    ["Low", "Elevated", "High"]
                ).all()
            )

            self.assertTrue(
                results["anomaly_flag"].isin([0, 1]).all()
            )

            saved_results = pd.read_csv(output_path)
            self.assertEqual(len(saved_results), 3)

    def test_non_finite_classifier_probabilities_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            probabilities = np.array([[np.nan, np.nan], [0.2, 0.8]])
            with self.assertRaisesRegex(ValueError, "non-finite probabilities"):
                self.run_inference_with_mocked_outputs(
                    temp_dir, probabilities=probabilities
                )

    def test_out_of_range_classifier_probabilities_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            probabilities = np.array([[0.2, 1.2], [0.2, 0.8]])
            with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
                self.run_inference_with_mocked_outputs(
                    temp_dir, probabilities=probabilities
                )

    def test_invalid_anomaly_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "invalid predictions"):
                self.run_inference_with_mocked_outputs(
                    temp_dir, anomaly_predictions=np.array([-1, 0])
                )

    def test_non_finite_anomaly_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "non-finite scores"):
                self.run_inference_with_mocked_outputs(
                    temp_dir, anomaly_scores=np.array([np.nan, 0.1])
                )

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "empty.csv"
            input_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                predict_customers(input_path, Path(temp_dir) / "out.csv")

    def test_oversized_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "large.csv"
            input_path.write_text("CONS_NO\\n1234567890", encoding="utf-8")

            with patch("src.inference.MAX_INPUT_FILE_BYTES", 5):
                with self.assertRaisesRegex(ValueError, "25 MB"):
                    predict_customers(
                        input_path,
                        Path(temp_dir) / "out.csv"
                    )

    def test_customer_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "customers.csv"
            input_path.write_text(
                "CONS_NO,reading\nCUSTOMER_1,1\nCUSTOMER_2,2\n",
                encoding="utf-8"
            )

            with patch("src.inference.MAX_CUSTOMERS", 1):
                with self.assertRaisesRegex(ValueError, "maximum allowed"):
                    predict_customers(
                        input_path,
                        Path(temp_dir) / "out.csv"
                    )

    def test_invalid_threshold_is_rejected(self):
        df = self.make_input(rows=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            output_path = Path(temp_dir) / "predictions.csv"

            df.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Threshold"):
                predict_customers(
                    input_path,
                    output_path,
                    threshold=0
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
