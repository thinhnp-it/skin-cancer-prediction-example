"""
Data Preprocessing Module
Handles data cleaning, encoding, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Diagnosis mapping dictionary
DX_DICT = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# Binary classification mapping
MALIGNANT_TYPES = ['mel', 'bcc', 'akiec', 'vasc']
BENIGN_TYPES = ['nv', 'df', 'bkl']


def handle_missing_values(df):
    """
    Handle missing values in the dataset

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame, SimpleImputer
        Processed DataFrame and fitted imputer
    """
    df_processed = df.copy()

    # Handle missing age values
    age_imputer = SimpleImputer(strategy='mean')
    if 'age' in df_processed.columns and df_processed['age'].isnull().any():
        df_processed['age'] = age_imputer.fit_transform(df_processed[['age']])
        print(f"Age missing values imputed with mean: {age_imputer.statistics_[0]:.2f}")

    # Handle missing categorical values
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            mode_value = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_value, inplace=True)
            print(f"{col} missing values filled with mode: {mode_value}")

    return df_processed, age_imputer


def create_diagnosis_labels(df):
    """
    Create diagnosis labels and binary classification labels

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with 'dx' column

    Returns:
    --------
    pd.DataFrame
        DataFrame with added diagnosis and binary label columns
    """
    df_processed = df.copy()

    # Create full diagnosis name
    df_processed['diagnosis'] = df_processed['dx'].map(DX_DICT)

    # Create binary label (0: Benign, 1: Malignant)
    df_processed['binary_label'] = df_processed['dx'].apply(
        lambda x: 1 if x in MALIGNANT_TYPES else 0
    )

    # Create binary label name
    df_processed['binary_class'] = df_processed['binary_label'].map({
        0: 'Benign',
        1: 'Malignant'
    })

    print("✓ Diagnosis labels created")
    return df_processed


def encode_categorical_features(df, categorical_columns=None):
    """
    Encode categorical variables using LabelEncoder

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    categorical_columns : list, optional
        List of columns to encode

    Returns:
    --------
    pd.DataFrame, dict
        Processed DataFrame and dictionary of label encoders
    """
    if categorical_columns is None:
        categorical_columns = ['sex', 'localization', 'dx_type']

    df_processed = df.copy()
    label_encoders = {}

    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            print(f"✓ Encoded {col}")

    # Encode diagnosis for multi-class classification
    le_diagnosis = LabelEncoder()
    df_processed['diagnosis_encoded'] = le_diagnosis.fit_transform(df_processed['dx'])
    label_encoders['diagnosis'] = le_diagnosis
    print("✓ Encoded diagnosis")

    return df_processed, label_encoders


def normalize_features(X):
    """
    Normalize features using StandardScaler

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix

    Returns:
    --------
    np.ndarray, StandardScaler
        Normalized features and fitted scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✓ Features normalized")
    return X_scaled, scaler


def prepare_binary_classification_data(df, feature_columns, test_size=0.2, random_state=42):
    """
    Prepare data for binary classification

    Parameters:
    -----------
    df : pd.DataFrame
        Processed DataFrame with binary labels
    feature_columns : list
        List of feature column names
    test_size : float
        Proportion of test set
    random_state : int
        Random seed

    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = df[feature_columns].copy()
    y = df['binary_label'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Binary classification split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def prepare_multiclass_classification_data(df, feature_columns, test_size=0.2, random_state=42):
    """
    Prepare data for multi-class classification

    Parameters:
    -----------
    df : pd.DataFrame
        Processed DataFrame with diagnosis labels
    feature_columns : list
        List of feature column names
    test_size : float
        Proportion of test set
    random_state : int
        Random seed

    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = df[feature_columns].copy()
    y = df['diagnosis_encoded'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Multi-class split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the module
    print("Preprocessing module test")
    print(f"DX_DICT: {DX_DICT}")
    print(f"Malignant types: {MALIGNANT_TYPES}")
    print(f"Benign types: {BENIGN_TYPES}")