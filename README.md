# Skin Cancer Detection - Capstone Project

## Project Overview

This project aims to build an intelligent system to support early diagnosis of skin cancer, especially melanoma, a dangerous type of skin cancer. The system assists dermatologists in evaluating skin images from patients, reducing diagnosis time and increasing early detection capability.

**Demo:** [Hugging Face Demo](https://huggingface.co/spaces/hoangkha1810/Skin_Cancer.Prediction_Cybersoft_Demo)

## Project Objectives

Build a **deep learning model** to classify skin lesions using actual skin lesion images from the HAM10000 dataset. The system uses Convolutional Neural Networks (CNNs) and transfer learning to classify:
- **Binary Classification:** Benign vs Malignant
- **Multi-class Classification:** 7 types of skin diseases

### Why Deep Learning with Images?

Using actual images instead of just metadata features provides:
- ✅ **Higher accuracy** (80-90%+ vs 60-70% with metadata only)
- ✅ **Rich visual features** (150K+ features vs 3 metadata features)
- ✅ **State-of-the-art performance** with transfer learning
- ✅ **Clinical relevance** - mimics how dermatologists diagnose

**Quick Start:** See [QUICK_START_IMAGES.md](QUICK_START_IMAGES.md) for a 3-step guide to get started!

## Project Structure

```
skin-cancer-detection/
├── venv/                  # Virtual environment (Python 3.11)
├── data/                  # Dataset directory
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/    # ~5,000 images
│   ├── HAM10000_images_part_2/    # ~5,000 images
│   └── processed/                  # Preprocessed data (auto-generated)
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

## Setup Instructions

### 1. Clone/Navigate to Project Directory

```bash
cd /Users/thinhnguyen/Documents/Data-AI-Class/ML_DL_02/buoi\ 10
```

### 2. Create Virtual Environment (Python 3.11)

```bash
python3.11 -m venv venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download Dataset

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

## Dataset Information

**Source:** ISIC - HAM10000 Dataset

**File:** HAM10000_metadata.csv

**Columns:**
- `image_id`: Image identifier
- `lesion_id`: Lesion identifier
- `dx`: Diagnosis code (mel, nv, bkl, bcc, akiec, vasc, df)
- `dx_type`: Diagnosis confirmation method
- `age`: Patient age
- `sex`: Patient gender
- `localization`: Body location of lesion

**Diagnosis Mapping (dx_dict):**
- `akiec`: Actinic keratoses and intraepithelial carcinoma
- `bcc`: Basal cell carcinoma
- `bkl`: Benign keratosis-like lesions
- `df`: Dermatofibroma
- `mel`: Melanoma
- `nv`: Melanocytic nevi
- `vasc`: Vascular lesions

---

## Technologies & Libraries

### Core Stack
- **Python 3.11**
- **Jupyter Notebook**

### Data Science & Visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical graphics
- **PIL (Pillow)**: Image loading and processing

### Deep Learning
- **TensorFlow/Keras**: Deep learning framework
  - CNN architectures
  - Transfer learning models (ResNet50, VGG16, MobileNet, EfficientNet)
  - Data augmentation (ImageDataGenerator)
  - Model training and evaluation

### Machine Learning (Utilities)
- **Scikit-learn**:
  - `train_test_split`: Dataset splitting
  - `LabelEncoder`: Label encoding
  - `confusion_matrix`: Model evaluation
  - `classification_report`: Performance metrics

### Deployment (Optional)
- **Streamlit**: Web interface deployment
- **FastAPI**: REST API development

---

## Best Practices

### 1. Clean Code
- Use clear, meaningful variable and function names
- Separate logic into small, reusable functions
- Add comments and documentation
- Implement error handling

### 2. Code Formatting
- Follow PEP 8 (Python Style Guide)
- Use `black` for automatic formatting
- Check code with `flake8` or `pylint`

### 3. Project Organization
- Maintain logical directory structure
- Use virtual environment for dependencies
- Implement version control (Git)

### 4. ML-Specific Standards
- Document data preprocessing steps
- Always split train/test sets properly
- Save models and parameters
- Track model performance metrics

### 5. Documentation & Reusability
- Create detailed README
- Save trained models for reuse
- Optimize code performance

---

## Expected Outcomes

1. **Complete Deep Learning Pipeline:** Image Preprocessing → CNN/Transfer Learning → Evaluation
2. **High Accuracy Models:** Deep learning models with 80-90%+ accuracy on skin lesion classification
3. **Understanding:** Deep knowledge of medical image classification and CNNs
4. **Practical Application:** Deployable model with Streamlit interface
5. **Extensible Codebase:** Clean, documented, reusable code for production use

---

## Next Steps

1. **Run Binary Classification:** Open [notebooks/Binary_Classification.ipynb](notebooks/Binary_Classification.ipynb) and run all cells
2. **Run Multi-class Classification:** Open [notebooks/Multiclass_Classification.ipynb](notebooks/Multiclass_Classification.ipynb) and run all cells
3. **Compare Results:** Analyze performance differences between binary and multi-class models
4. **Experiment:** Try different SVM parameters (C, gamma, kernel) to improve performance
5. **Deploy (Optional):** Create a Streamlit web application for predictions

---

## References

### Dataset & Research
- [ISIC Archive](https://challenge.isic-archive.com/)
- [HAM10000 Dataset Paper](https://arxiv.org/abs/1803.10417)
- [Kaggle HAM10000 Dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

### Deep Learning Resources
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

### General ML Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PEP 8 Style Guide](https://pep8.org/)

---

## License

This project is for educational purposes as part of the ML/DL course capstone.

---

## Contributors

- Thinh Nguyen

---

**Last Updated:** 2025-10-22