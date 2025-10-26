# Skin Cancer Detection - Capstone Project

## Project Overview

This project aims to build an intelligent system to support early diagnosis of skin cancer, especially melanoma, a dangerous type of skin cancer. The system assists dermatologists in evaluating skin images from patients, reducing diagnosis time and increasing early detection capability.

## Project Objectives

Build a **deep learning model** to classify skin lesions using actual skin lesion images from the HAM10000 dataset. The system uses Convolutional Neural Networks (CNNs) and transfer learning to classify:
- **Binary Classification:** Benign vs Malignant
- **Multi-class Classification:** 7 types of skin diseases

**Quick Start:** See [QUICK_START_IMAGES.md](QUICK_START_IMAGES.md) for a 3-step guide to get started!

## Project Structure

```
skin-cancer-detection/
├── venv/                  # Virtual environment (Python 3.11)
├── data/                  # Dataset directory
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/    # ~5,000 images
│   ├── HAM10000_images_part_2/    # ~5,000 images
├── notebooks/             # Main notebooks (simplified)
│   ├── Binary_Classification.ipynb      # Complete binary classification pipeline
│   └── Multiclass_Classification.ipynb  # Complete multi-class classification pipeline
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── image_loader.py    # Image loading & preprocessing utilities
│   ├── data_loader.py
│   └── preprocessing.py
├── models/                # Saved models (auto-generated)
├── results/               # Results, plots, metrics (auto-generated)
├── requirements.txt       # Project dependencies
└── README.md              # This file
```


### Download Dataset

Download the HAM10000 dataset from [ISIC Archive](https://challenge.isic-archive.com/data/) and place `HAM10000_metadata.csv` in the `data/` directory.

---

## Quick Start

### Option 1: Binary Classification (Benign vs Malignant)

Run the [Binary_Classification.ipynb](notebooks/Binary_Classification.ipynb) notebook to:
1. **Preprocess Data**: Load ~10,000 images, create binary labels, train/test split
2. **Train Model**: Build and train SVM classifier
3. **Evaluate Model**: Generate confusion matrix, ROC curve, and performance metrics

**Labels:**
- **Benign (0)**: nv, df, bkl
- **Malignant (1)**: mel, bcc, akiec, vasc

### Option 2: Multi-class Classification (7 Disease Types)

Run the [Multiclass_Classification.ipynb](notebooks/Multiclass_Classification.ipynb) notebook to:
1. **Preprocess Data**: Load images, create multi-class labels (7 diseases)
2. **Train Model**: Build and train multi-class SVM classifier
3. **Evaluate Model**: Generate confusion matrix, per-class metrics, and performance analysis

**Classes (7):**
- **nv**: Melanocytic nevi (~67%)
- **mel**: Melanoma (~11%)
- **bkl**: Benign keratosis-like lesions (~11%)
- **bcc**: Basal cell carcinoma (~5%)
- **akiec**: Actinic keratoses (~3%)
- **vasc**: Vascular lesions (~1%)
- **df**: Dermatofibroma (~1%)

---

## Notebook Structure

Each notebook follows a **3-step pipeline**:

### Step 1: Preprocess Data
- Load HAM10000 metadata
- Load and preprocess ~10,000 skin lesion images
- Resize to 224×224, normalize pixel values
- Create labels (binary or multi-class)
- Stratified train/test split (80/20)

### Step 2: Train Model
- Flatten images for SVM input
- Initialize SVM classifier with optimized parameters
- Train model on image data
- Display training progress and support vectors

### Step 3: Evaluate Model
- Calculate accuracy scores
- Generate confusion matrix
- Create classification report with precision/recall/F1
- Visualize ROC curves (binary) or per-class metrics (multi-class)
- Save model and results

---

## Expected Outcomes

1. **Complete Deep Learning Pipeline:** Image Preprocessing → CNN/Transfer Learning → Evaluation
2. **High Accuracy Models:** Deep learning models with 80-90%+ accuracy on skin lesion classification
3. **Understanding:** Deep knowledge of medical image classification and CNNs
4. **Practical Application:** Deployable model with Streamlit interface
5. **Extensible Codebase:** Clean, documented, reusable code for production use

---

## License

This project is for educational purposes as part of the ML/DL course capstone.

---

## Contributors

- Thinh Nguyen

---

**Last Updated:** 2025-10-22