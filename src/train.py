import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from model import UNet
from dataset import BrainTumorDataset
from losses import BCEDiceLoss
from metrics import dice_score, iou_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 5e-5

TRAIN_IMAGES_DIR = "dataset/images/train"
TRAIN_MASKS_DIR = "dataset/masks/train"

VAL_IMAGES_DIR = "dataset/images/val"
VAL_MASKS_DIR = "dataset/masks/val"

MODEL_SAVE_PATH = "models/unet_brain_tumor.pth"


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        batch_dice = dice_score(outputs, masks)
        batch_iou = iou_score(outputs, masks)

        total_loss += loss.item()
        total_dice += batch_dice
        total_iou += batch_iou

        loop.set_postfix(
            loss=loss.item(),
            dice=batch_dice,
            iou=batch_iou
        )

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def validate(model, dataloader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            batch_dice = dice_score(outputs, masks)
            batch_iou = iou_score(outputs, masks)

            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += batch_iou

            loop.set_postfix(
                loss=loss.item(),
                dice=batch_dice,
                iou=batch_iou
            )

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_dice, avg_iou


def main():
    print(f"Using device: {DEVICE}")

    Path("models").mkdir(exist_ok=True)

    train_dataset = BrainTumorDataset(
        images_dir=TRAIN_IMAGES_DIR,
        masks_dir=TRAIN_MASKS_DIR
    )

    val_dataset = BrainTumorDataset(
        images_dir=VAL_IMAGES_DIR,
        masks_dir=VAL_MASKS_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    best_val_dice = 0.0

    if Path(MODEL_SAVE_PATH).exists():
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_dice = checkpoint.get("best_val_dice", 0.0)

        print(f"Loaded previous model from {MODEL_SAVE_PATH}")
        print(f"Previous best Dice: {best_val_dice:.4f}")

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            loss_fn
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                    "epoch": epoch + 1,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE
                },
                MODEL_SAVE_PATH
            )

            print(f"Saved best model with Dice: {best_val_dice:.4f}")

    print("\nTraining finished.")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()