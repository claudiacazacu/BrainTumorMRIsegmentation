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
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {
        background-color: #2e3d57;
        color: #e8ecf2;
        font-family: 'Sora', sans-serif;
    }

    [data-testid="stHeader"] { background-color: transparent; }
    [data-testid="stAppViewContainer"] { padding-top: 0; }
    [data-testid="stMainBlockContainer"] { padding-top: 0; }

    .hero {
        background: linear-gradient(135deg, #1a2540 0%, #2d3f60 60%, #3a4f72 100%);
        border-radius: 16px;
        padding: 2.8rem 2.5rem 2.4rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.09);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(100,140,200,0.15) 0%, transparent 70%);
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 300px; height: 150px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(80,120,180,0.1) 0%, transparent 70%);
    }
    .hero h1 {
        margin: 0 0 0.6rem 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .hero p {
        margin: 0 0 1.4rem 0;
        font-size: 0.92rem;
        color: rgba(255,255,255,0.45);
        font-weight: 300;
        max-width: 540px;
    }
    .hero-links {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: rgba(255,255,255,0.6);
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
    }
    .hero-github {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        text-decoration: none !important;
        transition: background 0.2s;
    }
    .hero-github:hover {
        background: rgba(255,255,255,0.15);
        color: white !important;
    }

    [data-testid="stFileUploader"] {
        background-color: rgba(15, 20, 38, 0.45);
        border: 1.5px dashed rgba(255,255,255,0.13);
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
    }

    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.3);
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin: 1.8rem 0;
    }

    .result-card {
        background: rgba(15, 20, 38, 0.5);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.07);
        height: 100%;
    }
    .result-card .card-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.38);
        margin-bottom: 0.7rem;
    }

    .metric-wrap {
        background: rgba(15, 20, 38, 0.5);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-wrap .metric-title {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 0.6rem;
    }
    .metric-wrap .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e05555;
        line-height: 1;
    }
    .metric-wrap .metric-sub {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.28);
        margin-top: 0.35rem;
        font-weight: 300;
    }
    .metric-wrap .metric-note {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.2);
        margin-top: 1.2rem;
        font-weight: 300;
        line-height: 1.5;
    }

    .app-footer {
        text-align: center;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.35);
        margin-top: 3rem;
        padding: 1.2rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.07);
        font-weight: 300;
    }
    .app-footer a {
        color: rgba(255,255,255,0.45) !important;
        text-decoration: none;
    }
    .app-footer a:hover {
        color: rgba(255,255,255,0.75) !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>Brain Tumor MRI Segmentation</h1>
    <p>Upload a brain MRI scan to automatically detect and segment the tumor region using a U-Net deep learning model.</p>
    <div class="hero-links">
        <span class="hero-tag">U-Net</span>
        <span class="hero-tag">PyTorch</span>
        <span class="hero-tag">Binary Segmentation</span>
        <a class="hero-github" href="https://github.com/claudiacazacu/BrainTumorMRIsegmentation" target="_blank">
            &#9679; GitHub
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


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

    tumor_pct = float(np.sum(mask_np) / mask_np.size * 100)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Segmentation Results</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([4, 4, 4, 3], gap="medium")

    with col1:
        st.markdown('<div class="result-card"><div class="card-label">Original MRI</div>', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card"><div class="card-label">Predicted Mask</div>', unsafe_allow_html=True)
        st.image(predicted_mask_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="result-card"><div class="card-label">Overlay</div>', unsafe_allow_html=True)
        st.image(overlay_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-wrap">
            <div class="metric-title">Tumor Area</div>
            <div class="metric-value">{tumor_pct:.1f}%</div>
            <div class="metric-sub">of total scan area</div>
            <div class="metric-note">Percentage is computed relative to the full image area, not the brain region alone.</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="app-footer">
    Brain Tumor MRI Segmentation &nbsp;·&nbsp; Claudia Cazacu &nbsp;·&nbsp; University of Bucharest &nbsp;·&nbsp; 2026
    &nbsp;·&nbsp;
    <a href="https://github.com/claudiacazacu/BrainTumorMRIsegmentation" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
