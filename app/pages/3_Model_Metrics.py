import streamlit as st

from app.components.model_utils import (
    read_classifier_reports,
    read_detection_report,
)


st.title("Model metrics")
st.write(
    "Evaluation metrics from the reports currently available in this checkout."
)

st.subheader("Classification")
try:
    classifier_reports = read_classifier_reports()
except (OSError, ValueError) as error:
    st.error(f"Could not read classification reports: {error}")
else:
    if classifier_reports:
        st.dataframe(classifier_reports, use_container_width=True)
        chart_rows = {
            f"{row['model']} · {row['configuration']}": row["test_accuracy"]
            for row in classifier_reports
            if row["test_accuracy"] is not None
        }
        if chart_rows:
            st.bar_chart(
                {"Test accuracy": chart_rows},
                y_label="Accuracy",
                x_label="Model and configuration",
            )
    else:
        st.info(
            "No classifier reports found. Train or evaluate the classifiers "
            "to generate reports under reports/config*/."
        )

st.subheader("Object detection")
try:
    detection_report = read_detection_report()
except (OSError, ValueError) as error:
    st.error(f"Could not read the detection report: {error}")
else:
    if detection_report is None:
        st.info(
            "No detection test report found at "
            "reports/detection/test_metrics.json."
        )
    else:
        metric_columns = st.columns(4)
        for column, label, key in zip(
            metric_columns,
            ("Precision", "Recall", "mAP@50", "mAP@50–95"),
            ("precision", "recall", "map50", "map50_95"),
        ):
            value = detection_report.get(key)
            column.metric(label, f"{value:.1%}" if value is not None else "—")

        per_class = detection_report.get("per_class_map50_95", {})
        if per_class:
            st.caption("Detection mAP@50–95 by class")
            st.bar_chart(
                {"mAP@50–95": per_class},
                y_label="mAP@50–95",
                x_label="Class",
            )