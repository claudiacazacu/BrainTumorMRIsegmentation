import streamlit as st

from utils import (
    load_model,
    preprocess_uploaded_image,
    predict_mask,
    create_overlay,
    mask_to_image,
)


st.set_page_config(
    page_title="Brain Tumor MRI Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("Brain Tumor MRI Segmentation")

@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError:
    st.error("Model file not found: models/unet_brain_tumor.pth")
    st.stop()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

THRESHOLD = 0.5

if uploaded_file is not None:
    original_image, image_tensor = preprocess_uploaded_image(uploaded_file)

    mask_np = predict_mask(
        model=model,
        image_tensor=image_tensor,
        threshold=THRESHOLD
    )

    predicted_mask_image = mask_to_image(mask_np)
    overlay_image = create_overlay(original_image, mask_np)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original MRI")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Predicted Mask")
        st.image(predicted_mask_image, use_container_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay_image, use_container_width=True)
else:
    st.info("Upload an MRI image to start.")