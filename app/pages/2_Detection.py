import streamlit as st

from app.components.image_viewer import read_uploaded_image
from app.components.model_utils import detect_objects


st.title("Brain tumor detection")
st.write(
    "Upload an MRI image to find and visualize regions detected by the trained "
    "YOLO model."
)
st.warning(
    "Detections are for research and demonstration only, not medical advice."
)

confidence_threshold = st.slider(
    "Minimum detection confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Predictions below this confidence threshold are hidden.",
)
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    key="detection_image",
)

if uploaded_file is not None:
    try:
        image = read_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
    else:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
        if st.button("Detect tumor regions", type="primary"):
            try:
                annotated_image, detections = detect_objects(
                    image,
                    confidence_threshold,
                )
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
                st.error(f"Could not run detection: {error}")
            else:
                st.subheader("Detection results")
                st.image(
                    annotated_image,
                    caption="Detected regions",
                    use_container_width=True,
                )
                if detections:
                    st.dataframe(detections, use_container_width=True)
                    st.write(f"Found **{len(detections)}** detection(s).")
                else:
                    st.info(
                        "No regions met the selected confidence threshold."
                    )