# Brain Tumor MRI Segmentation using U-Net

This project focuses on brain tumor segmentation from MRI images.
The goal is to identify the tumor region in an MRI scan and generate a binary segmentation mask.

## Technologies Used

- **Python** - core programming language used throughout the project
- **PyTorch** - deep learning framework used to build and train the segmentation model
- **U-Net** - convolutional neural network architecture used for the segmentation task
- **Streamlit** - used to build the web interface for uploading images and viewing predictions
- **NumPy** - numerical operations and array handling
- **Pillow** - image loading and processing
- **Matplotlib** - plotting training curves and result figures
- **tqdm** - progress bars during training and evaluation

## Dataset

The dataset used is **BRISC2025**, available on Kaggle.
Only the segmentation part of the dataset was used.

## Features

- MRI image preprocessing with optional data augmentation
- U-Net model for tumor segmentation
- Training and validation using PyTorch
- Experiment tracking - each training run saved separately in `experiments/`
- Streamlit interface for uploading MRI images
- Display of the original image, predicted mask, and overlay result
- Tumor area percentage calculated from the predicted mask

## Data Augmentation

Applied during training only (not validation). Transformations include:
- Horizontal and vertical flips
- Random rotation (90°, 180°, 270°)
- Brightness adjustment (±20%)
- Gaussian noise

## Evaluation Metrics

**Dice Score** measures how well the predicted mask overlaps with the real tumor mask.
A higher Dice Score means better segmentation.

**IoU Score** measures the overlap between the predicted mask and the ground truth compared to their total combined area.
A higher IoU means better prediction quality.

## Results

Across the four training runs (baseline, higher LR, augmented 20ep, augmented 40ep):

```text
Validation Dice Score: 0.83 - 0.86  (best: 0.8569, augmented 40ep)
Validation IoU Score:  0.73 - 0.76  (best: 0.7629, augmented 40ep)
```

## Training

```bash
conda activate <your_env>
cd BrainTumorMRIsegmentation
python src/train.py
```

Each run is saved in `experiments/run_<timestamp>/` and contains:
- `combined_metrics.png` - Loss, Dice, IoU train vs. val in one figure
- `metrics_table.png` - final metrics table
- `training_history.json` / `training_history.csv`
- `best_model.pth`
- `config.json`

## Comparing Experiments

After multiple training runs:

```bash
python src/compare_experiments.py
```

Generates in `outputs/`:
- `experiments_comparison_table.png` - side-by-side metrics for all runs
- `experiments_comparison_curves.png` - overlaid val Dice and IoU curves

## Project Structure

```
BrainTumorMRIsegmentation/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── masks/
│       ├── train/
│       ├── val/
│       └── test/
├── experiments/         # one folder per training run
├── models/              # latest best model (used by Streamlit)
├── outputs/             # comparison charts
├── src/
│   ├── train.py
│   ├── compare_experiments.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── preprocess.py
│   └── predict.py
└── app/                 # Streamlit interface
```
