import streamlit as st

from app.components.image_viewer import read_uploaded_image
from app.components.model_utils import (
    CLASSIFICATION_MODELS,
    classify_image,
)


st.title("Brain tumor classification")
st.write(
    "Upload an MRI image to estimate probabilities for the four classes. "
    "The image is resized to 224 × 224 pixels before inference."
)
st.warning(
    "This prediction is for research and demonstration only, not medical advice."
)

model_name = st.selectbox("Classification model", list(CLASSIFICATION_MODELS))
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    key="classification_image",
)

if uploaded_file is not None:
    try:
        image = read_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
    else:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
        if st.button("Classify image", type="primary"):
            try:
                predictions = classify_image(image, model_name)
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
                st.error(f"Could not run classification: {error}")
            else:
                predicted_class, confidence = predictions[0]
                st.subheader("Prediction")
                st.metric("Most likely class", predicted_class)
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                st.subheader("Class probabilities")
                st.bar_chart(
                    {
                        "Probability": {
                            class_name: score
                            for class_name, score in predictions
                        }
                    },
                    y_label="Probability",
                    x_label="Class",
                )