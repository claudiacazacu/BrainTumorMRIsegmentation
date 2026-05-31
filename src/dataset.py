from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment

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

    def _augment(self, image, mask):
        """Aplica aceleasi transformari random pe imagine SI masca."""

        # Horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Vertical flip
        if random.random() < 0.3:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Rotatie 90/180/270 grade
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Brightness (doar pe imagine, nu pe masca)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0.0, 1.0)

        # Gaussian noise (doar pe imagine)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self._find_mask(image_path)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask