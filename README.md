# GridGuard AI
### Smart Electricity Theft Risk Detection & Consumption Anomaly Monitoring

GridGuard AI is a machine-learning project that analyzes electricity consumption patterns to identify customers whose usage profiles may warrant further investigation.

It combines supervised classification with unsupervised anomaly detection and provides an interactive Streamlit dashboard for model evaluation, risk review, anomaly monitoring, and scoring new customer data.

> **Project status:** Research and portfolio prototype. A high-risk prediction is an investigation signal, not proof of electricity theft. This is not a production utility-decision system.

## Key Features

- Customer-level electricity consumption feature engineering.
- Random Forest classifier for theft-risk prediction.
- Isolation Forest for consumption anomaly detection.
- Validation-based classification threshold selection.
- Training-only median imputation.
- Customer-level stratified train, validation, and test splits.
- Interactive dashboard with six sections:
  - Overview
  - Investigation Queue
  - Anomaly Monitoring
  - Model Evaluation
  - Feature Importance
  - New Data Scoring
- Input validation for malformed data, duplicate IDs, missing readings, oversized files, and excessive customer counts.
- CSV prediction export.

## Technology Stack

Python, pandas, NumPy, scikit-learn, Joblib, Streamlit, Plotly, and unittest.

## Dataset

The project uses the State Grid Corporation of China (SGCC) electricity-theft dataset.

| Dataset property | Value |
|---|---:|
| Customers | 42,372 |
| Normal customers | 38,757 |
| Theft-labelled customers | 3,615 |
| Daily consumption readings | 1,034 |
| Positive-label proportion | 8.53% |

The raw dataset is not included in this repository. Obtain it from the [original research repository](https://github.com/henryRDlab/ElectricityTheftDetection/) and follow its applicable terms.

## Machine Learning Pipeline

### Feature Engineering

The pipeline extracts 23 customer-level features from daily consumption readings:

- Consumption statistics: mean, median, standard deviation, minimum, maximum, range, quartiles, and interquartile range.
- Consumption behavior: zero-reading ratio and coefficient of variation.
- Temporal patterns: eight period-level averages.
- Recent consumption: first and last 30-day averages.
- Data completeness: valid reading count and missing ratio.

### Data Splitting

Customers are split using stratified sampling and a fixed random seed.

| Partition | Customers |
|---|---:|
| Training | 25,423 |
| Validation | 8,474 |
| Test | 8,475 |

The median imputer is fitted on training data only. The validation set is used for threshold selection, while the test set is reserved for final evaluation.

### Supervised Classification

A Random Forest classifier is trained with class balancing to address label imbalance. Its decision threshold is selected on the validation set using the F2 score.

### Anomaly Detection

An Isolation Forest model is trained on normal training customers to identify unusual consumption patterns independently of the supervised classifier.

## Test Set Results

Results on the held-out test partition using the validation-selected classification threshold:

| Metric | Result |
|---|---:|
| Accuracy | 77.44% |
| Precision | 21.35% |
| Recall | 61.27% |
| F1-score | 31.67% |
| ROC-AUC | 0.7804 |
| PR-AUC | 0.3187 |

### Confusion Matrix

| | Predicted normal | Predicted theft |
|---|---:|---:|
| Actual normal | 6,120 | 1,632 |
| Actual theft | 280 | 443 |

The model identifies a meaningful portion of theft-labelled customers, but its false-positive count is substantial. Predictions should support human investigation rather than automatic enforcement.

## Dashboard

Launch the dashboard from the project root:

```powershell
python -m streamlit run .\app\app.py
```

Dashboard sections:

1. **Overview** — dataset summary and headline evaluation metrics.
2. **Investigation Queue** — risk-ranked customers and filtering.
3. **Anomaly Monitoring** — unusual consumption patterns.
4. **Model Evaluation** — test metrics and confusion matrix.
5. **Feature Importance** — Random Forest feature importance.
6. **New Data Scoring** — upload customer consumption data and export predictions.

## Score New Customer Data

Run inference from the repository root:

```powershell
python -m src.inference path\to\customer_data.csv --output reports\predictions.csv
```

To override the decision threshold:

```powershell
python -m src.inference path\to\customer_data.csv --output reports\predictions.csv --threshold 0.20
```

Input data must contain the customer ID column and the expected consumption date columns defined in `models/consumption_schema.json`.

The output includes:

| Column | Description |
|---|---|
| customer_id | Customer identifier |
| theft_probability | Predicted probability |
| predicted_label | Binary classification |
| risk_level | High, Elevated, or Low |
| anomaly_flag | Isolation Forest anomaly indicator |
| anomaly_score | Anomaly decision score |

## Installation

Python 3.11 or a compatible version is recommended.

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the original dataset in the expected raw-data location before running training:

```powershell
.\.venv\Scripts\python.exe .\src\train.py
```

Launch the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app\app.py
```

## Tests

Run the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover feature extraction, schema validation, customer ID validation, inference behavior, threshold validation, customer limits, and invalid model outputs.

## Project Structure

```text
GridGuard-AI/
├── app/
│   └── app.py
├── data/
│   └── raw/                 # Downloaded dataset; not committed
├── models/
│   ├── anomaly_detector.joblib
│   ├── consumption_schema.json
│   ├── feature_columns.joblib
│   ├── feature_imputer.joblib
│   └── theft_classifier.joblib
├── reports/
│   ├── feature_importance.csv
│   ├── metrics.json
│   └── test_predictions.csv
├── src/
│   ├── features.py
│   ├── inference.py
│   └── train.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
└── requirements.txt
```

## Limitations

- Dataset labels are used as provided and are not independently verified theft findings.
- False positives remain substantial and require human review.
- Feature importance does not establish causal relationships.
- Probability calibration has not been independently established.
- The project has not been validated in a live electricity distribution environment.
- Production deployment, operational monitoring, and utility-system integration are outside the current scope.

## Research Reference

Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids." *IEEE Transactions on Industrial Informatics*, 14(4), 1606–1615.

https://doi.org/10.1109/TII.2017.2785963

## Author

Developed as an applied machine-learning and smart-grid analytics portfolio project.