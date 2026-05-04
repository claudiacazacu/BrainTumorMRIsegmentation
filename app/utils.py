import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from model import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = ROOT_DIR / "models" / "unet_brain_tumor.pth"
IMG_SIZE = (256, 256)


@torch.no_grad()
def load_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


def preprocess_uploaded_image(uploaded_file):
    original_image = Image.open(uploaded_file).convert("L")

    resized_image = original_image.resize(IMG_SIZE)
    image_np = np.array(resized_image, dtype=np.float32) / 255.0

    image_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    return original_image, image_tensor


@torch.no_grad()
def predict_mask(model, image_tensor, threshold=0.5):
    logits = model(image_tensor)
    probabilities = torch.sigmoid(logits)

    mask = (probabilities > threshold).float()
    mask_np = mask.squeeze().cpu().numpy()

    return mask_np


def mask_to_image(mask_np):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    return mask_image


def create_overlay(original_image, mask_np):
    original_rgb = original_image.convert("RGB")
    image_np = np.array(original_rgb).astype(np.uint8)

    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image = mask_image.resize(original_image.size)

    mask_resized = np.array(mask_image) / 255.0
    tumor_pixels = mask_resized > 0.5

    overlay = image_np.copy()

    red_mask = np.zeros_like(image_np)
    red_mask[:, :, 0] = 255

    alpha = 0.45

    overlay[tumor_pixels] = (
        (1 - alpha) * overlay[tumor_pixels]
        + alpha * red_mask[tumor_pixels]
    ).astype(np.uint8)

    return Image.fromarray(overlay)