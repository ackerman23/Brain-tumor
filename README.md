# MRI Brain Tumor Detection

A deep learning model to detect brain tumors from MRI scans using PyTorch.

## Overview

This project implements a Convolutional Neural Network (CNN) to classify brain MRI scans into two categories:
- Healthy brain scans
- Brain scans with tumors

## Features

- Image preprocessing and resizing to 128x128 pixels
- Data augmentation for better model generalization
- Custom PyTorch Dataset implementation
- Training/Validation split
- Real-time loss tracking and visualization
- Binary classification (tumor/no tumor)

## Requirements

- Python 3.x
- PyTorch
- NumPy
- OpenCV (cv2)
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install torch numpy opencv-python matplotlib scikit-learn
```

## Dataset

The project uses the Brain MRI Images for Brain Tumor Detection dataset from Kaggle:
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

Dataset structure:
```
data/
└── brain_tumor_dataset/
    ├── yes/    # Contains MRI scans with tumors
    └── no/     # Contains MRI scans without tumors
```

## Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional layers
- Max pooling layers
- Fully connected layers
- Binary classification output

## Usage

1. Download the dataset from Kaggle and place it in the `data/` directory
2. Open and run the Jupyter notebook `MRI-Brain-Tumor-Detector.ipynb`
3. The notebook includes:
   - Data loading and preprocessing
   - Model training
   - Loss visualization
   - Model evaluation

## Results

The model tracks both training and validation loss throughout the training process, allowing for:
- Monitoring of model convergence
- Detection of overfitting
- Performance evaluation


