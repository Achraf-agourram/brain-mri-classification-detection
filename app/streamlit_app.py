import streamlit as st


st.set_page_config(
    page_title="Brain MRI AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Brain MRI analysis")
st.markdown(
    """
    Explore trained brain MRI models using an image you upload. Choose a
    classifier to estimate the tumor category, or run the object detector to
    locate visible tumor regions.

    **For research and demonstration only.** Model predictions are not a
    diagnosis and must not replace advice from a qualified medical professional.
    """
)

st.subheader("Choose an analysis")
classification_column, detection_column = st.columns(2)

with classification_column:
    st.markdown("### Image classification")
    st.write("Compare MobileNetV2 and ResNet50 tumor-category predictions.")
    st.page_link(
        "pages/1_Classification.py",
        label="Open classification",
        icon=":material/category:",
    )

with detection_column:
    st.markdown("### Tumor detection")
    st.write("Run YOLO on an MRI image and inspect predicted bounding boxes.")
    st.page_link(
        "pages/2_Detection.py",
        label="Open detection",
        icon=":material/center_focus_strong:",
    )

st.page_link(
    "pages/3_Model_Metrics.py",
    label="View model metrics",
    icon=":material/monitoring:",
)