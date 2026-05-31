"""
compare_experiments.py
----------------------
Citeste toate runurile din experiments/ si genereaza:
  - outputs/experiments_comparison_table.png  — tabel comparativ cu metrici finale
  - outputs/experiments_comparison_curves.png — grafic suprapus val Dice + val IoU pentru toate runurile

Rulare:
    python src/compare_experiments.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS_DIR = Path("experiments")
OUTPUTS_DIR = Path("outputs")


def load_runs():
    """Incarca config + training history din fiecare run folder."""
    runs = []

    for run_folder in sorted(EXPERIMENTS_DIR.iterdir()):
        if not run_folder.is_dir():
            continue

        history_path = run_folder / "training_history.json"
        config_path = run_folder / "config.json"

        if not history_path.exists():
            continue

        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        best = max(history, key=lambda r: r["val_dice"])
        last = history[-1]

        runs.append({
            "name": run_folder.name,
            "config": config,
            "history": history,
            "best": best,
            "last": last,
        })

    return runs


def save_comparison_table(runs, output_path):
    """Salveaza un tabel PNG cu metrici finale pentru fiecare run."""
    headers = ["Run", "LR", "Batch", "Epochs", "Best Val Dice", "Best Val IoU", "Final Val Loss"]

    rows = []
    for r in runs:
        cfg = r["config"]
        best = r["best"]
        rows.append([
            r["name"],
            str(cfg.get("learning_rate", "—")),
            str(cfg.get("batch_size", "—")),
            str(cfg.get("epochs", "—")),
            f"{best['val_dice']:.4f}",
            f"{best['val_iou']:.4f}",
            f"{r['last']['val_loss']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.55 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    table.auto_set_column_width(col=list(range(len(headers))))

    # Header styling
    for col in range(len(headers)):
        table[(0, col)].set_facecolor("#2c7bb6")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    # Highlight best val dice row
    best_idx = max(range(len(rows)), key=lambda i: float(rows[i][4]))
    for col in range(len(headers)):
        table[(best_idx + 1, col)].set_facecolor("#d4edda")

    # Alternating row colors
    for row_idx in range(len(rows)):
        if row_idx == best_idx:
            continue
        color = "#f9f9f9" if row_idx % 2 == 0 else "#ffffff"
        for col in range(len(headers)):
            table[(row_idx + 1, col)].set_facecolor(color)

    ax.set_title("Experiments Comparison — Model Performance Metrics", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison table: {output_path}")


def save_comparison_curves(runs, output_path):
    """Salveaza un grafic suprapus cu val Dice si val IoU pentru toate runurile."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Validation Metrics Across Experiments", fontsize=14, fontweight="bold")

    for r in runs:
        history = r["history"]
        epochs = [row["epoch"] for row in history]
        label = r["name"]

        axes[0].plot(epochs, [row["val_dice"] for row in history], label=label)
        axes[1].plot(epochs, [row["val_iou"] for row in history], label=label)

    axes[0].set_title("Validation Dice Score")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_title("Validation IoU Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU Score")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison curves: {output_path}")


def main():
    runs = load_runs()

    if not runs:
        print("No experiment runs found in experiments/")
        return

    print(f"Found {len(runs)} run(s): {[r['name'] for r in runs]}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    save_comparison_table(runs, OUTPUTS_DIR / "experiments_comparison_table.png")
    save_comparison_curves(runs, OUTPUTS_DIR / "experiments_comparison_curves.png")

    print("\nDone. Files saved in outputs/")


if __name__ == "__main__":
    main()
