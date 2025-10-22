"""
Data Loading Module
Handles loading and basic validation of HAM10000 dataset
"""

import pandas as pd
import os


def load_ham10000_metadata(file_path='../data/HAM10000_metadata.csv'):
    """
    Load HAM10000 metadata CSV file

    Parameters:
    -----------
    file_path : str
        Path to HAM10000_metadata.csv

    Returns:
    --------
    pd.DataFrame
        Loaded metadata DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

    return df


def validate_dataset(df):
    """
    Validate dataset structure

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate

    Returns:
    --------
    bool
        True if validation passes
    """
    required_columns = ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")

    print("✓ Dataset validation passed")
    return True


if __name__ == "__main__":
    # Test the module
    df = load_ham10000_metadata()
    validate_dataset(df)
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")