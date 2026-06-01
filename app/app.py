import numpy as np
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
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    /* Background & base */
    .stApp {
        background-color: #3d4f70;
        color: #f0f2f6;
    }

    [data-testid="stHeader"] {
        background-color: rgba(45, 57, 82, 0.95);
    }

    /* Sidebar-style top bar */
    .app-header {
        background: linear-gradient(135deg, #2d3952 0%, #4a5d80 100%);
        border-radius: 12px;
        padding: 1.8rem 2rem 1.4rem 2rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.01em;
    }
    .app-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        color: rgba(255,255,255,0.55);
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: rgba(25, 30, 45, 0.5);
        border: 1.5px dashed rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }

    /* Result cards */
    .result-card {
        background: rgba(25, 30, 50, 0.55);
        border-radius: 10px;
        padding: 1rem 1rem 0.6rem 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-card h4 {
        margin: 0 0 0.6rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(255,255,255,0.6);
    }

    /* Metric badge */
    .metric-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e05555 0%, #b03030 100%);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        margin-top: 0.4rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        margin-top: 0.3rem;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.5rem 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.3);
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 Brain Tumor MRI Segmentation</h1>
    <p>Upload an MRI scan to detect and segment the tumor region using a trained U-Net model.</p>
</div>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────
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


# ── Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drag and drop or browse an MRI image (JPG, PNG, TIF)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    label_visibility="visible",
)

THRESHOLD = 0.5

if uploaded_file is not None:
    original_image, image_tensor = preprocess_uploaded_image(uploaded_file)
    mask_np = predict_mask(model=model, image_tensor=image_tensor, threshold=THRESHOLD)
    predicted_mask_image = mask_to_image(mask_np)
    overlay_image = create_overlay(original_image, mask_np)

    # Tumor area %
    tumor_pct = float(np.sum(mask_np) / mask_np.size * 100)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Metric row ───────────────────────────────────────
    _, metric_col, _ = st.columns([3, 2, 3])
    with metric_col:
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="metric-label">Estimated Tumor Area</div>
            <div class="metric-badge">{tumor_pct:.2f}% of scan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Image columns ────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="result-card"><h4>Original MRI</h4>', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card"><h4>Predicted Mask</h4>', unsafe_allow_html=True)
        st.image(predicted_mask_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="result-card"><h4>Overlay</h4>', unsafe_allow_html=True)
        st.image(overlay_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Brain Tumor MRI Segmentation &nbsp;·&nbsp; Claudia Cazacu &nbsp;·&nbsp; University of Bucharest &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
