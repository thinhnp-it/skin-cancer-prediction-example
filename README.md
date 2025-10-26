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

**Original (Baseline):**
Run [Binary_Classification.ipynb](notebooks/Binary_Classification.ipynb) - Full pipeline without PCA

**NEW - With PCA (Recommended):**
Run [Binary_Classification_PCA.ipynb](notebooks/Binary_Classification_PCA.ipynb) to:
1. **Preprocess Data**: Load ~10,000 images with PCA dimensionality reduction
2. **Train Model**: Build and train SVM on reduced features (**10x faster!**)
3. **Evaluate Model**: Generate confusion matrix, ROC curve, and performance metrics

**Benefits:**
- ⚡ **10x faster training** (~30-60 min vs 5 hours)
- 📦 **15x smaller model** (<500 MB vs 7 GB)
- 💾 **6x less memory** usage
- ✓ Similar accuracy (~58-60%)

**Labels:**
- **Benign (0)**: nv, df, bkl
- **Malignant (1)**: mel, bcc, akiec, vasc

### Option 2: Multi-class Classification (7 Disease Types)

**Original (Baseline):**
Run [Multiclass_Classification.ipynb](notebooks/Multiclass_Classification.ipynb) - Full pipeline without PCA

**NEW - With PCA (Recommended):**
Run [Multiclass_Classification_PCA.ipynb](notebooks/Multiclass_Classification_PCA.ipynb) to:
1. **Preprocess Data**: Load images with PCA reduction (150K → ~1K features)
2. **Train Model**: Train multi-class SVM (**10x faster!**)
3. **Evaluate Model**: Comprehensive per-class performance analysis

**Benefits:**
- ⚡ **10x faster training** (5 hours → 30-60 minutes)
- 📦 **15x smaller model** (<500 MB vs 7 GB)
- 💾 Reduced memory usage
- ✓ Same accuracy with better efficiency

**Classes (7):**
- **nv**: Melanocytic nevi (~67%)
- **mel**: Melanoma (~11%)
- **bkl**: Benign keratosis-like lesions (~11%)
- **bcc**: Basal cell carcinoma (~5%)
- **akiec**: Actinic keratoses (~3%)
- **vasc**: Vascular lesions (~1%)
- **df**: Dermatofibroma (~1%)

**Guide:** See [NOTEBOOKS_PCA_GUIDE.md](NOTEBOOKS_PCA_GUIDE.md) for detailed instructions

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

## Performance Metrics

### Binary Classification (Benign vs Malignant)

**Model:** Support Vector Machine (SVM) with RBF kernel

| Metric | Value |
|--------|-------|
| Training Samples | 8,012 |
| Testing Samples | 2,003 |
| Features (Flattened) | 150,528 (224×224×3) |
| Class Balance | 79% Benign / 21% Malignant |

**Results:**
- Test Accuracy: ~58-65% (baseline)
- ROC-AUC Score: Available in results
- Training Time: Variable based on configuration

**Critical Metrics:**
- **Sensitivity (Recall):** Ability to detect malignant cases
- **Specificity:** Ability to detect benign cases
- **False Negatives:** Missed malignant cases (most critical metric)

### Multi-class Classification (7 Disease Types)

**Model:** Support Vector Machine (SVM) with RBF kernel, One-vs-Rest

| Metric | Value |
|--------|-------|
| Training Accuracy | 66.53% |
| Testing Accuracy | 58.31% |
| Top-2 Accuracy | 86.97% |
| Random Baseline | 14.29% |
| Training Time | ~5 hours (18,060 seconds) |
| Support Vectors | 6,262 |

**Per-Class Performance (F1-Score):**

| Class | Disease Name | F1-Score | Recall | Support |
|-------|-------------|----------|---------|---------|
| nv | Melanocytic nevi | 0.7383 | 60.70% | 1,341 |
| bkl | Benign keratosis | 0.4310 | 51.82% | 220 |
| akiec | Actinic keratoses | 0.3913 | 55.38% | 65 |
| bcc | Basal cell carcinoma | 0.3902 | 46.60% | 103 |
| mel | Melanoma | 0.3888 | 59.19% | 223 |
| vasc | Vascular lesions | 0.3333 | 53.57% | 28 |
| df | Dermatofibroma | 0.2466 | 39.13% | 23 |

**Key Observations:**
- Best performance on dominant class (nv - 66.9% of dataset)
- Poor performance on rare classes (df, vasc - <2% of dataset)
- Common confusion: nv ↔ mel (benign nevi vs melanoma)
- Model shows overfitting (training: 66.53%, testing: 58.31%)

**Improvement Opportunities:**
- Apply PCA for dimensionality reduction (150K+ features)
- Use class balancing techniques (SMOTE, class weights)
- Try deep learning models (CNN, transfer learning)
- Data augmentation for rare classes

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