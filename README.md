# Skin Cancer Detection - Capstone Project

## Project Overview

This project aims to build an intelligent system to support early diagnosis of skin cancer, especially melanoma, a dangerous type of skin cancer. The system assists dermatologists in evaluating skin images from patients, reducing diagnosis time and increasing early detection capability.

## Project Objectives

Build a **ML model** to classify skin lesions using actual skin lesion images from the HAM10000 dataset.
- **Binary Classification:** Benign vs Malignant
- **Multi-class Classification:** 7 types of skin diseases

## Project Structure

```
skin-cancer-detection/
├── venv/                  # Virtual environment (Python 3.11)
├── data/                  # Dataset directory - HAM10000 dataset
├── notebooks/             # Main notebooks (simplified)
├── src/                   # Source code modules
├── models/                # Saved models (auto-generated)
├── results/               # Results, plots, metrics (auto-generated)
├── requirements.txt       # Project dependencies
└── README.md              # This file
```


### Download Dataset

Download the HAM10000 dataset from [ISIC Archive](https://challenge.isic-archive.com/data/) and place `HAM10000_metadata.csv` in the `data/` directory.

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