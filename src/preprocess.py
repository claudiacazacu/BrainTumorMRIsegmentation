from pathlib import Path
import shutil
import random

from PIL import Image
from tqdm import tqdm


RAW_DIR = Path("dataset_raw")
OUT_DIR = Path("dataset")

IMG_SIZE = (256, 256)
VAL_SPLIT = 0.2
SEED = 42

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def recreate_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_image_files(folder: Path):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def preprocess_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path):
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    image = image.resize(IMG_SIZE)
    mask = mask.resize(IMG_SIZE)

    mask = mask.point(lambda p: 255 if p > 0 else 0)

    image.save(out_image)
    mask.save(out_mask)


def process_split(pairs, split_name):
    image_out_dir = OUT_DIR / "images" / split_name
    mask_out_dir = OUT_DIR / "masks" / split_name

    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in tqdm(pairs, desc=f"Processing {split_name}"):
        out_image = image_out_dir / image_path.name
        out_mask = mask_out_dir / mask_path.name

        preprocess_pair(image_path, mask_path, out_image, out_mask)


def get_pairs(split_name):
    image_dir = RAW_DIR / split_name / "images"
    mask_dir = RAW_DIR / split_name / "masks"

    images = find_image_files(image_dir)
    masks = find_image_files(mask_dir)

    mask_dict = {m.stem: m for m in masks}

    pairs = []

    for image_path in images:
        image_name = image_path.stem

        if image_name in mask_dict:
            pairs.append((image_path, mask_dict[image_name]))
        else:
            print(f"[WARNING] No mask found for {image_path.name}")

    return pairs

def main():
    random.seed(SEED)

    recreate_dir(OUT_DIR / "images")
    recreate_dir(OUT_DIR / "masks")

    train_pairs = get_pairs("train")
    test_pairs = get_pairs("test")

    random.shuffle(train_pairs)

    val_size = int(len(train_pairs) * VAL_SPLIT)
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]

    print(f"Train images: {len(train_pairs)}")
    print(f"Val images: {len(val_pairs)}")
    print(f"Test images: {len(test_pairs)}")

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")
    process_split(test_pairs, "test")

    print("Preprocessing finished successfully.")


if __name__ == "__main__":
    main()