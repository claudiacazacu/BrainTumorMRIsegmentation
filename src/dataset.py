from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.images = self._find_images(self.images_dir)

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def _find_images(self, folder):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(folder.glob(f"*{ext}"))
            files.extend(folder.glob(f"*{ext.upper()}"))
        return sorted(files)

    def _find_mask(self, image_path):
        for ext in IMAGE_EXTENSIONS:
            mask_path = self.masks_dir / f"{image_path.stem}{ext}"
            if mask_path.exists():
                return mask_path

            mask_path_upper = self.masks_dir / f"{image_path.stem}{ext.upper()}"
            if mask_path_upper.exists():
                return mask_path_upper

        raise FileNotFoundError(f"No mask found for image: {image_path.name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self._find_mask(image_path)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        mask = (mask > 0.5).astype(np.float32)

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask