import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))

from model import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/unet_brain_tumor.pth"

IMAGE_PATH = "dataset/images/test/brisc2025_test_00906_pi_co_t1.jpg"
MASK_PATH = "dataset/masks/test/brisc2025_test_00906_pi_co_t1.png"

OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    image = Image.open(image_path).convert("L")
    image_np = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)
    return image, tensor


def main():
    print(f"Using device: {DEVICE}")

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image, image_tensor = load_image(IMAGE_PATH)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        TRESHOLD = 0.5
        pred_mask = (prob > TRESHOLD).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    output_mask = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
    output_mask.save(OUTPUT_DIR / "predicted_mask.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI Image")
    axes[0].axis("off")

    if Path(MASK_PATH).exists():
        mask = Image.open(MASK_PATH).convert("L")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
    else:
        axes[1].text(0.5, 0.5, "No ground truth", ha="center")
        axes[1].set_title("Ground Truth Mask")

    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(pred_mask_np, alpha=0.5)
    axes[2].set_title("Predicted Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_result.png")
    plt.show()

    print(f"Saved prediction mask to: {OUTPUT_DIR / 'predicted_mask.png'}")
    print(f"Saved visualization to: {OUTPUT_DIR / 'prediction_result.png'}")


if __name__ == "__main__":
    main()