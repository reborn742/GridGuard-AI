import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import MAX_INPUT_FILE_BYTES, predict_customers


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


st.set_page_config(
    page_title="GridGuard AI | Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 16px;
        border-radius: 10px;
    }
    .hero {
        padding: 22px 24px;
        border-radius: 12px;
        background: linear-gradient(120deg, #102b46, #174e68);
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 5px;
    }
    .hero p {
        color: #d4e8f2;
        margin-bottom: 0;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    with open(REPORTS / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)

    predictions = pd.read_csv(REPORTS / "test_predictions.csv")
    importance = pd.read_csv(REPORTS / "feature_importance.csv")

    required_predictions = {
        "customer_id",
        "actual_label",
        "theft_probability",
        "predicted_label",
        "anomaly_flag"
    }

    if not required_predictions.issubset(predictions.columns):
        raise ValueError("Test prediction report has an invalid schema.")

    if predictions.empty:
        raise ValueError("Test prediction report is empty.")

    return metrics, predictions, importance


try:
    metrics, predictions, importance = load_data()
except Exception as exc:
    st.error(f"Unable to load project reports: {exc}")
    st.stop()


test = metrics["test"]
anomaly = test["anomaly_detection"]
default_threshold = float(metrics["selected_threshold"])

# Fixed evaluation threshold: never tune against the held-out test set.
historical_risk = predictions["theft_probability"] >= default_threshold

total_customers = len(predictions)
flagged_customers = int(historical_risk.sum())
actual_positive = int(predictions["actual_label"].sum())


with st.sidebar:
    st.title("⚡ GridGuard AI")
    st.caption("Electricity Theft Risk Intelligence")

    st.divider()

    st.subheader("Evaluation Configuration")
    st.metric("Selected threshold", f"{default_threshold:.4f}")
    st.caption("Selected using validation data (F2).")

    st.metric("Test customers", f"{total_customers:,}")
    st.metric("Test flagged", f"{flagged_customers:,}")

    st.divider()

    st.caption("Model architecture")
    st.write("**Supervised:** Random Forest")
    st.write("**Unsupervised:** Isolation Forest")
    st.write("**Features:** 23 statistical and temporal")

    st.divider()
    st.caption("Research prototype | Investigation support only")


st.markdown("""
<div class="hero">
    <h1>GridGuard AI</h1>
    <p>Smart Electricity Theft Risk Detection & Consumption Anomaly Monitoring</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "A risk score indicates unusual patterns that may warrant investigation. "
    "It does not establish or confirm electricity theft."
)


overview, investigation, anomalies, evaluation, features, scoring = st.tabs([
    "Overview",
    "Investigation Queue",
    "Anomaly Monitoring",
    "Model Evaluation",
    "Feature Importance",
    "New Data Scoring"
])


# -------------------------------------------------------------------
# OVERVIEW
# -------------------------------------------------------------------

with overview:
    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Customers", f"{metrics['dataset_customers']:,}")
    c2.metric("Training Customers", f"{metrics['train_customers']:,}")
    c3.metric("Test Customers", f"{metrics['test_customers']:,}")
    c4.metric("Theft-labelled Test", f"{actual_positive:,}")

    st.divider()

    st.subheader("Held-out Test Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Recall", f"{test['recall']:.2%}")
    c2.metric("Precision", f"{test['precision']:.2%}")
    c3.metric("F1 Score", f"{test['f1']:.2%}")
    c4.metric("PR-AUC", f"{test['pr_auc']:.4f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Reference Label Distribution")

        label_data = pd.DataFrame({
            "Category": ["Normal-labelled", "Theft-labelled"],
            "Customers": [
                int((predictions["actual_label"] == 0).sum()),
                int((predictions["actual_label"] == 1).sum())
            ]
        })

        fig = px.pie(
            label_data,
            names="Category",
            values="Customers",
            hole=0.58
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Predicted Risk Distribution")

        fig = px.histogram(
            predictions,
            x="theft_probability",
            nbins=40,
            title="Customer Risk Probability"
        )
        fig.add_vline(
            x=default_threshold,
            line_dash="dash",
            annotation_text="Selected threshold"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Training Information")

    a, b, c = st.columns(3)
    a.metric("Engineered Features", metrics["feature_count"])
    b.metric("Validation Customers", f"{metrics['validation_customers']:,}")
    c.metric("Training Runtime", f"{metrics['runtime_seconds']} sec")


# -------------------------------------------------------------------
# INVESTIGATION QUEUE
# -------------------------------------------------------------------

with investigation:
    st.subheader("Customer Investigation Queue")

    st.caption(
        "This queue uses the fixed validation-selected threshold. "
        "Ground-truth labels are excluded from operational records."
    )

    queue = predictions.loc[historical_risk].copy()

    queue["risk_level"] = queue["theft_probability"].apply(
        lambda value: "High" if value >= 0.75 else "Elevated"
    )

    queue = queue.sort_values(
        "theft_probability",
        ascending=False
    )

    f1, f2, f3 = st.columns([2, 1, 1])

    with f1:
        search = st.text_input(
            "Search customer ID",
            placeholder="Enter customer ID..."
        )

    with f2:
        risk_filter = st.selectbox(
            "Risk level",
            ["All", "High", "Elevated"]
        )

    with f3:
        anomaly_filter = st.selectbox(
            "Anomaly status",
            ["All", "Anomaly only", "No anomaly"]
        )

    filtered = queue.copy()

    if search:
        filtered = filtered[
            filtered["customer_id"].astype(str).str.contains(
                search, case=False, regex=False
            )
        ]

    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    if anomaly_filter == "Anomaly only":
        filtered = filtered[filtered["anomaly_flag"] == 1]
    elif anomaly_filter == "No anomaly":
        filtered = filtered[filtered["anomaly_flag"] == 0]

    high_count = int((queue["risk_level"] == "High").sum())
    elevated_count = int((queue["risk_level"] == "Elevated").sum())

    a, b, c = st.columns(3)
    a.metric("Total Flagged", f"{len(queue):,}")
    b.metric("High Risk", f"{high_count:,}")
    c.metric("Elevated Risk", f"{elevated_count:,}")

    st.write(f"Showing **{len(filtered):,}** matching records.")

    display = filtered[[
        "customer_id",
        "theft_probability",
        "risk_level",
        "anomaly_flag"
    ]].rename(columns={
        "customer_id": "Customer ID",
        "theft_probability": "Risk Probability",
        "risk_level": "Risk Level",
        "anomaly_flag": "Anomaly Flag"
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Probability": st.column_config.ProgressColumn(
                format="%.2f",
                min_value=0,
                max_value=1
            ),
            "Anomaly Flag": st.column_config.CheckboxColumn()
        }
    )

    st.download_button(
        "Download Investigation Queue",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="gridguard_investigation_queue.csv",
        mime="text/csv",
        disabled=display.empty
    )


# -------------------------------------------------------------------
# ANOMALY MONITORING
# -------------------------------------------------------------------

with anomalies:
    st.subheader("Unsupervised Anomaly Monitoring")

    a, b, c = st.columns(3)

    a.metric("Anomaly Flags", f"{anomaly['flagged_customers']:,}")
    b.metric("Precision", f"{anomaly['precision']:.2%}")
    c.metric("Recall", f"{anomaly['recall']:.2%}")

    st.caption(
        "Isolation Forest was trained on normal-labelled training customers. "
        "Evaluation below compares its flags against dataset reference labels."
    )

    st.write(f"**F1 Score:** {anomaly['f1']:.2%}")

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            predictions,
            x="theft_probability",
            y="anomaly_flag",
            color="actual_label",
            opacity=0.55,
            labels={
                "theft_probability": "Theft Risk Probability",
                "anomaly_flag": "Anomaly Flag",
                "actual_label": "Reference Label"
            },
            title="Risk Probability vs Anomaly Flag"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = predictions["anomaly_flag"].value_counts().rename(
            index={0: "Normal profile", 1: "Anomaly"}
        ).reset_index()

        counts.columns = ["Profile", "Customers"]

        fig = px.bar(
            counts,
            x="Profile",
            y="Customers",
            title="Anomaly Detection Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "An unusual consumption profile is not proof of electricity theft. "
        "Anomaly flags should be assessed alongside other evidence."
    )


# -------------------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------------------

with evaluation:
    st.subheader("Final Held-out Test Evaluation")

    a, b, c = st.columns(3)

    a.metric("ROC-AUC", f"{test['roc_auc']:.4f}")
    b.metric("PR-AUC", f"{test['pr_auc']:.4f}")
    c.metric("Validation Threshold", f"{default_threshold:.4f}")

    a, b, c, d = st.columns(4)

    a.metric("Precision", f"{test['precision']:.2%}")
    b.metric("Recall", f"{test['recall']:.2%}")
    c.metric("F1", f"{test['f1']:.2%}")
    d.metric("Accuracy", f"{test['accuracy']:.2%}")

    st.subheader("Confusion Matrix")

    matrix = pd.DataFrame(
        test["confusion_matrix"],
        index=["Actual Normal", "Actual Theft-labelled"],
        columns=["Predicted Normal", "Predicted Flagged"]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title="Held-out Test Confusion Matrix",
        labels={
            "x": "Prediction",
            "y": "Reference Label",
            "color": "Customers"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Threshold Comparison")

    baseline = test["baseline_at_0_50"]

    comparison = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1"],
        "Validation-selected": [
            test["precision"],
            test["recall"],
            test["f1"]
        ],
        "Default 0.50": [
            baseline["precision"],
            baseline["recall"],
            baseline["f1"]
        ]
    })

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.caption(
        "The threshold was selected using validation data. "
        "The held-out test set was not used for threshold tuning."
    )


# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------

with features:
    st.subheader("Model Feature Importance")

    st.caption(
        "Random Forest impurity-based feature importance. "
        "These values indicate model reliance, not causality."
    )

    top = importance.head(15).sort_values("importance")

    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Features"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Feature Importance",
        data=importance.to_csv(index=False).encode("utf-8"),
        file_name="gridguard_feature_importance.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------------
# NEW DATA SCORING
# -------------------------------------------------------------------

with scoring:
    st.subheader("New Customer Data Scoring")

    st.write(
        "Upload a CSV containing customer IDs and the same 1,034 dated "
        "consumption columns used during model training."
    )

    st.info(
        "The FLAG target column is not required. "
        "The input schema is validated before prediction."
    )

    uploaded = st.file_uploader(
        "Upload customer consumption CSV",
        type=["csv"],
        key="new_customer_upload"
    )

    if uploaded is not None and uploaded.size > MAX_INPUT_FILE_BYTES:
        st.error("File exceeds the 25 MB upload limit. Please upload a smaller CSV.")
        uploaded = None

    operational_threshold = st.slider(
        "Investigation threshold",
        min_value=0.01,
        max_value=0.95,
        value=default_threshold,
        step=0.01,
        help=(
            "This controls operational flags for new records only. "
            "It does not modify or recalculate historical test metrics."
        )
    )

    if uploaded is not None:
        st.caption(
            f"Uploaded file: {uploaded.name} | "
            f"{uploaded.size / (1024 * 1024):.2f} MB"
        )

        if st.button("Run Customer Risk Analysis", type="primary"):
            try:
                with st.spinner("Validating records and generating predictions..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir = Path(temp_dir)

                        input_file = temp_dir / "uploaded_customers.csv"
                        output_file = temp_dir / "predictions.csv"

                        input_file.write_bytes(uploaded.getvalue())

                        result = predict_customers(
                            input_file,
                            output_file,
                            operational_threshold
                        )

                        st.session_state["latest_scoring_results"] = result
                        st.success("Customer scoring completed.")

            except Exception as exc:
                st.error(f"Scoring failed: {exc}")

    result = st.session_state.get("latest_scoring_results")

    if result is not None:
        st.divider()
        st.subheader("Scoring Results")

        a, b, c, d = st.columns(4)

        a.metric("Customers", f"{len(result):,}")
        b.metric("Flagged", f"{int(result['predicted_label'].sum()):,}")
        c.metric("High Risk", f"{int((result['risk_level'] == 'High').sum()):,}")
        d.metric("Anomaly Flags", f"{int(result['anomaly_flag'].sum()):,}")

        st.dataframe(
            result.sort_values(
                "theft_probability",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download New Customer Predictions",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="gridguard_new_customer_predictions.csv",
            mime="text/csv"
        )


st.divider()

st.caption(
    "GridGuard AI | SGCC Electricity Theft Detection Dataset | "
    "Random Forest + Isolation Forest | Research and decision-support system"
)
