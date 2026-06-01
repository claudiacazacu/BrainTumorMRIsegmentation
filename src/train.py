import sys
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from model import UNet
from dataset import BrainTumorDataset
from losses import BCEDiceLoss
from metrics import dice_score, iou_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 1e-4

TRAIN_IMAGES_DIR = "dataset/images/train"
TRAIN_MASKS_DIR = "dataset/masks/train"

VAL_IMAGES_DIR = "dataset/images/val"
VAL_MASKS_DIR = "dataset/masks/val"

FINAL_MODEL_PATH = Path("models/unet_brain_tumor.pth")
EXPERIMENTS_DIR = Path("experiments")


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


def create_run_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = EXPERIMENTS_DIR / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def save_config(run_folder, image_size):
    config = {
        "project": "Brain Tumor MRI Segmentation",
        "model": "U-Net",
        "framework": "PyTorch",
        "loss_function": "BCE + Dice Loss",
        "optimizer": "Adam",
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "augmentation": True,
        "image_size": image_size,
        "train_images_dir": TRAIN_IMAGES_DIR,
        "train_masks_dir": TRAIN_MASKS_DIR,
        "val_images_dir": VAL_IMAGES_DIR,
        "val_masks_dir": VAL_MASKS_DIR
    }

    with open(run_folder / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)


def save_history(run_folder, history):
    csv_path = run_folder / "training_history.csv"
    json_path = run_folder / "training_history.json"

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_dice",
        "val_dice",
        "train_iou",
        "val_iou"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def save_plots(run_folder, history):
    epochs = [row["epoch"] for row in history]

    LW = 2.5  # line width

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle("Training vs Validation Metrics", fontsize=11, fontweight="bold")

    # Loss
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train", color="#1f77b4", linewidth=LW)
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val", color="#ff7f0e", linestyle="--", linewidth=LW)
    axes[0].set_xlabel("Epoch", fontsize=8)
    axes[0].set_ylabel("Loss", fontsize=8)
    axes[0].set_title("Loss", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(True, linewidth=0.5)

    # Dice
    axes[1].plot(epochs, [row["train_dice"] for row in history], label="Train", color="#1f77b4", linewidth=LW)
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Val", color="#ff7f0e", linestyle="--", linewidth=LW)
    axes[1].set_xlabel("Epoch", fontsize=8)
    axes[1].set_ylabel("Dice Score", fontsize=8)
    axes[1].set_title("Dice Score", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(True, linewidth=0.5)

    # IoU
    axes[2].plot(epochs, [row["train_iou"] for row in history], label="Train", color="#1f77b4", linewidth=LW)
    axes[2].plot(epochs, [row["val_iou"] for row in history], label="Val", color="#ff7f0e", linestyle="--", linewidth=LW)
    axes[2].set_xlabel("Epoch", fontsize=8)
    axes[2].set_ylabel("IoU Score", fontsize=8)
    axes[2].set_title("IoU Score", fontsize=9)
    axes[2].legend(fontsize=7)
    axes[2].tick_params(labelsize=7)
    axes[2].grid(True, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(run_folder / "combined_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Keep individual plots for backward compatibility
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_folder / "loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_dice"] for row in history], label="Train Dice")
    plt.plot(epochs, [row["val_dice"] for row in history], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_folder / "dice.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_iou"] for row in history], label="Train IoU")
    plt.plot(epochs, [row["val_iou"] for row in history], label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Training and Validation IoU Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_folder / "iou.png")
    plt.close()


def save_metrics_table(run_folder, history):
    """Save a PNG table with final epoch metrics (best val dice epoch + last epoch)."""
    best = max(history, key=lambda r: r["val_dice"])
    last = history[-1]

    rows = [
        ["Metric", "Best Val Dice Epoch", "Final Epoch"],
        ["Epoch", str(best["epoch"]), str(last["epoch"])],
        ["Train Loss", f"{best['train_loss']:.4f}", f"{last['train_loss']:.4f}"],
        ["Val Loss", f"{best['val_loss']:.4f}", f"{last['val_loss']:.4f}"],
        ["Train Dice", f"{best['train_dice']:.4f}", f"{last['train_dice']:.4f}"],
        ["Val Dice", f"{best['val_dice']:.4f}", f"{last['val_dice']:.4f}"],
        ["Train IoU", f"{best['train_iou']:.4f}", f"{last['train_iou']:.4f}"],
        ["Val IoU", f"{best['val_iou']:.4f}", f"{last['val_iou']:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Header row styling
    for col in range(3):
        table[(0, col)].set_facecolor("#2c7bb6")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    # Highlight best val dice row (row index 4 = Val Dice)
    for col in range(3):
        table[(5, col)].set_facecolor("#d4edda")

    ax.set_title("Model Performance Metrics", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(run_folder / "metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print(f"Using device: {DEVICE}")
    print("Starting training from scratch.")

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    run_folder = create_run_folder()
    best_model_path = run_folder / "best_model.pth"

    print(f"Experiment folder: {run_folder}")

    train_dataset = BrainTumorDataset(
        images_dir=TRAIN_IMAGES_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        augment=True
    )

    val_dataset = BrainTumorDataset(
        images_dir=VAL_IMAGES_DIR,
        masks_dir=VAL_MASKS_DIR,
        augment=False
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

    sample_image, _ = train_dataset[0]
    image_size = list(sample_image.shape[1:])

    save_config(run_folder, image_size)

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = 0.0
    history = []

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

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
            "train_iou": train_iou,
            "val_iou": val_iou
        })

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

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_dice": best_val_dice,
                "epoch": epoch + 1,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE
            }

            torch.save(checkpoint, best_model_path)
            shutil.copy2(best_model_path, FINAL_MODEL_PATH)

            print(f"Saved best model with Dice: {best_val_dice:.4f}")

    save_history(run_folder, history)
    save_plots(run_folder, history)
    save_metrics_table(run_folder, history)

    print("\nTraining finished.")
    print(f"Experiment folder: {run_folder.resolve()}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model path: {best_model_path.resolve()}")
    print(f"Streamlit model path: {FINAL_MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()