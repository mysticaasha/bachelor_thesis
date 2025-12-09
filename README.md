# Satellite Image Classification with YOLO and EuroSAT Dataset

This repository contains the code and resources for a project that uses the YOLO (You Only Look Once) deep learning model to classify satellite images from the EuroSAT land cover dataset. The project aims to identify various types of land cover using multispectral satellite imagery.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model and Methods](#model-and-methods)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)

## Project Overview
The goal of this project is to classify land cover types from satellite images using the YOLO model. We utilized the EuroSAT dataset, which contains multispectral images captured by the Sentinel-2 satellite. This project demonstrates the potential of deep learning models like YOLO for remote sensing applications.

## Dataset
- **EuroSAT Dataset**: A dataset that contains 13-channel multispectral images covering 10 different land cover classes. Each image is 64x64 pixels.
- **Classes**: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, Sea/Lake.
- [EuroSAT Dataset](https://github.com/phelber/EuroSAT)  

## Model and Methods
- **YOLO (You Only Look Once)**: A state-of-the-art object detection and classification model known for its speed and accuracy. We use the YOLOv8 version for this project.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce the 13-channel multispectral images to 3 channels to make them compatible with YOLO.
- **Deep Learning Framework**: The YOLO model was implemented using the Ultralytics library, which provides a streamlined interface for training and evaluation.

## Preprocessing
1. **Data Splitting**: The dataset was split into training, validation, and test sets.
2. **PCA Application**: Applied PCA to reduce the dimensionality of images from 13 to 3 channels.
3. **Normalization**: Images were normalized to fit the input format required by the YOLO model.
4. **Format Conversion**: Converted images into a format compatible with YOLO.

## Training
- The YOLOv8 model was trained on the processed EuroSAT dataset.
- **Hyperparameters**: Configured for optimal performance on the dataset.
  - Epochs: 100
  - Image size: 64x64 pixels
  - Device: CPU (Change to GPU if available for faster training)
  
## Evaluation
- The model was evaluated using accuracy metrics:
  - **Top-1 Accuracy**: Measures how often the top predicted label is correct.
  - **Top-5 Accuracy**: Measures how often the correct label is within the top 5 predictions.
- Additional metrics such as confusion matrix and inference speed were also used to assess performance.

## Results
- The trained YOLO model achieved significant accuracy on the EuroSAT dataset, demonstrating the effectiveness of deep learning for land cover classification in satellite imagery.
<img width="700" height="90" alt="image" src="https://github.com/user-attachments/assets/9320acba-5786-442c-a1a7-354ac9da2d95" />
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/464da0b7-a56f-4d27-80da-9f247f127f4e" />

## Requirements
- Python 3.8+
- Libraries: 
  - numpy
  - rasterio
  - scikit-learn
  - tqdm
  - Pillow
  - ultralytics (for YOLOv8)
- Install requirements via:
  ```bash
  pip install -r requirements.txt

## Directory Structure
- ├── data/
- |   ├── original_data/       # Raw EuroSAT data
- |   ├── pretrained_data/     # Processed data ready for YOLO
- ├── models/
- |   ├── yolov8n-cls.pt       # YOLO model weights
- ├── main.py
- ├── requirements.txt         # Required libraries
