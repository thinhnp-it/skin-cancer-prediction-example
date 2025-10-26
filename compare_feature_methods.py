"""
Compare Different Feature Extraction Methods

This script compares the performance of different feature extraction methods:
1. Flattened pixels (baseline)
2. HOG features (handcrafted)
3. VGG16 features (deep learning) # TODO


Usage:
    python compare_feature_methods.py
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import time

from image_loader import load_images_from_metadata
from preprocessing import prepare_images_with_feature_extraction, prepare_images_for_training

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("FEATURE EXTRACTION METHOD COMPARISON")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\nLoading data...")

metadata_path = 'data/HAM10000_metadata.csv'
df = pd.read_csv(metadata_path)

# Create binary labels
malignant_types = ['mel', 'bcc', 'akiec', 'vasc']
df['binary_label'] = df['dx'].apply(lambda x: 1 if x in malignant_types else 0)

# Load images
print("Loading images...")
images, loaded_image_ids = load_images_from_metadata(
    df,
    base_path='data',
    target_size=(224, 224),
    normalize=True,
    verbose=True
)

# Split data
df_filtered = df[df['image_id'].isin(loaded_image_ids)].reset_index(drop=True)
y_binary = df_filtered['binary_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    images, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print(f"\nData split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ============================================================================
# Define Methods to Compare
# ============================================================================

methods = {
    'Flattened Pixels + PCA': {
        'type': 'baseline',
        'params': {'use_pca': True, 'variance_ratio': 0.95}
    },
    'HOG Features': {
        'type': 'feature',
        'params': {
            'method': 'hog',
            'target_size': (128, 128),
            'use_pca': False
        }
    }
}

# ============================================================================
# Compare Methods
# ============================================================================

results = []

for method_name, config in methods.items():
    print(f"\n{'='*80}")
    print(f"Method: {method_name}")
    print(f"{'='*80}")

    # Preprocessing
    start_time = time.time()

    if config['type'] == 'baseline':
        prep_data = prepare_images_for_training(X_train, X_test, **config['params'])
    else:
        prep_data = prepare_images_with_feature_extraction(X_train, X_test, **config['params'])

    prep_time = time.time() - start_time

    # Training
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42,
              probability=True, class_weight='balanced')

    train_start = time.time()
    svm.fit(prep_data['X_train_final'], y_train)
    train_time = time.time() - train_start

    # Evaluation
    y_train_pred = svm.predict(prep_data['X_train_final'])
    y_test_pred = svm.predict(prep_data['X_test_final'])
    y_test_proba = svm.predict_proba(prep_data['X_test_final'])[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    # Store results
    result = {
        'Method': method_name,
        'Feature Dim': prep_data['X_train_final'].shape[1],
        'Prep Time (s)': prep_time,
        'Train Time (s)': train_time,
        'Total Time (s)': prep_time + train_time,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'ROC-AUC': roc_auc,
        'Support Vectors': sum(svm.n_support_)
    }
    results.append(result)

    print(f"\nResults:")
    print(f"  Feature dimensions: {result['Feature Dim']:,}")
    print(f"  Preprocessing time: {result['Prep Time (s)']:.2f}s")
    print(f"  Training time: {result['Train Time (s)']:.2f}s")
    print(f"  Test accuracy: {result['Test Acc']:.4f} ({result['Test Acc']*100:.2f}%)")
    print(f"  ROC-AUC: {result['ROC-AUC']:.4f}")

# ============================================================================
# Create Comparison Table
# ============================================================================

print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}\n")

df_results = pd.DataFrame(results)

# Format table
print(df_results.to_string(index=False))

# ============================================================================
# Visualize Comparison
# ============================================================================

os.makedirs('results/comparison', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Test Accuracy
axes[0, 0].bar(df_results['Method'], df_results['Test Acc'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 0].set_ylabel('Test Accuracy', fontsize=12)
axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['Test Acc']):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 2: ROC-AUC
axes[0, 1].bar(df_results['Method'], df_results['ROC-AUC'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 1].set_ylabel('ROC-AUC Score', fontsize=12)
axes[0, 1].set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['ROC-AUC']):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 3: Training Time
axes[1, 0].bar(df_results['Method'], df_results['Total Time (s)'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_ylabel('Total Time (seconds)', fontsize=12)
axes[1, 0].set_title('Processing + Training Time', fontsize=14, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['Total Time (s)']):
    axes[1, 0].text(i, v + max(df_results['Total Time (s)'])*0.02, f'{v:.1f}s', ha='center', fontweight='bold')

# Plot 4: Feature Dimensions
axes[1, 1].bar(df_results['Method'], df_results['Feature Dim'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 1].set_ylabel('Number of Features', fontsize=12)
axes[1, 1].set_title('Feature Dimensionality', fontsize=14, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['Feature Dim']):
    axes[1, 1].text(i, v * 1.2, f'{v:,}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('results/comparison/feature_method_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nComparison plot saved to: results/comparison/feature_method_comparison.png")

# Save results
df_results.to_csv('results/comparison/feature_comparison_results.csv', index=False)
print(f"Results table saved to: results/comparison/feature_comparison_results.csv")

# ============================================================================
# Recommendations
# ============================================================================

print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

best_acc_idx = df_results['Test Acc'].idxmax()
best_speed_idx = df_results['Total Time (s)'].idxmin()
best_auc_idx = df_results['ROC-AUC'].idxmax()

print(f"\nBest Test Accuracy: {df_results.loc[best_acc_idx, 'Method']}")
print(f"  → Accuracy: {df_results.loc[best_acc_idx, 'Test Acc']:.4f}")
print(f"  → ROC-AUC: {df_results.loc[best_acc_idx, 'ROC-AUC']:.4f}")

print(f"\nFastest Training: {df_results.loc[best_speed_idx, 'Method']}")
print(f"  → Total time: {df_results.loc[best_speed_idx, 'Total Time (s)']:.2f}s")
print(f"  → Accuracy: {df_results.loc[best_speed_idx, 'Test Acc']:.4f}")

print(f"\nBest ROC-AUC: {df_results.loc[best_auc_idx, 'Method']}")
print(f"  → ROC-AUC: {df_results.loc[best_auc_idx, 'ROC-AUC']:.4f}")
print(f"  → Accuracy: {df_results.loc[best_auc_idx, 'Test Acc']:.4f}")

print(f"\n{'='*80}")
print("COMPARISON COMPLETE!")
print(f"{'='*80}")
