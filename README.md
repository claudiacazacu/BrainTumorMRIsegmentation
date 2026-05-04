# Brain Tumor MRI Segmentation using U-Net

This project focuses on brain tumor segmentation from MRI images.  
The goal is to identify the tumor region in an MRI scan and generate a binary segmentation mask.

## Technologies Used

- Python
- PyTorch
- U-Net
- Streamlit
- NumPy
- Pillow

## Dataset

The dataset used is **BRISC2025**, available on Kaggle.  
Only the segmentation part of the dataset was used.

## Features

- MRI image preprocessing
- U-Net model for tumor segmentation
- Training and validation using PyTorch
- Streamlit interface for uploading MRI images
- Display of the original image, predicted mask, and overlay result

## Evaluation Metrics

**Dice Score** measures how well the predicted mask overlaps with the real tumor mask.  
A higher Dice Score means better segmentation.

**IoU Score** measures the overlap between the predicted mask and the ground truth compared to their total combined area.  
A higher IoU means better prediction quality.

## Results

The model achieved approximately:

```text
Validation Dice Score: 0.85 - 0.86
Validation IoU Score: 0.75 - 0.76