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
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #576A8F;
            color: white;
        }

        [data-testid="stHeader"] {
            background-color: rgba(87, 106, 143, 0.85);
        }

        [data-testid="stFileUploader"] {
            background-color: rgba(31, 35, 43, 0.65);
            padding: 1rem;
            border-radius: 10px;
        }

        [data-testid="stVerticalBlock"] {
            background-color: transparent;
        }

        footer {
            display: inline-block;
            background-color: rgba(31, 35, 43, 0.55);
            padding: 0.45rem 0.9rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-top: 2rem;
        }

        footer p {
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

THRESHOLD = 0.5

if uploaded_file is not None:
    original_image, image_tensor = preprocess_uploaded_image(uploaded_file)

    mask_np = predict_mask(
        model=model,
        image_tensor=image_tensor,
        threshold=THRESHOLD,
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


st.markdown(
    """
    <div style="text-align: center;">
        <footer>
            <p>Proiect ML 2026<br>Claudia Cazacu</p>
        </footer>
    </div>
    """,
    unsafe_allow_html=True,
)