"""
Data Preprocessing Module
Handles data cleaning, encoding, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')


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


def flatten_images(X):
    """Flatten an image array of shape (N, H, W, C) or (N, H, W) to (N, -1).

    Parameters:
    -----------
    X : np.ndarray
        Image array with leading batch dimension.

    Returns:
    --------
    np.ndarray
        Flattened feature matrix of shape (N, D).
    """
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)


def flatten_and_scale_images(X_train, X_test, *, scaler=None):
    """Flatten train/test image arrays and standardize features with StandardScaler.

    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Image arrays (N, H, W[, C])
    scaler : sklearn.preprocessing.StandardScaler or None
        If provided, will be used to transform data; otherwise a new scaler is fitted on X_train.

    Returns:
    --------
    tuple: (X_train_flat, X_test_flat, X_train_scaled, X_test_scaled, scaler)
    """
    X_train_flat = flatten_images(X_train)
    X_test_flat = flatten_images(X_test)

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
    else:
        X_train_scaled = scaler.transform(X_train_flat)

    X_test_scaled = scaler.transform(X_test_flat)

    return X_train_flat, X_test_flat, X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train_scaled, X_test_scaled, n_components=None, variance_ratio=0.95, pca=None):
    """Apply PCA for dimensionality reduction.

    Parameters:
    -----------
    X_train_scaled, X_test_scaled : np.ndarray
        Scaled feature matrices
    n_components : int or None
        Number of components to keep. If None, uses variance_ratio.
    variance_ratio : float
        Minimum cumulative variance ratio to retain (used if n_components is None)
    pca : sklearn.decomposition.PCA or None
        If provided, will be used to transform data; otherwise a new PCA is fitted on X_train_scaled.

    Returns:
    --------
    tuple: (X_train_pca, X_test_pca, pca, explained_variance_ratio)
    """
    if pca is None:
        if n_components is None:
            # Find optimal number of components based on variance ratio
            pca = PCA(n_components=variance_ratio, svd_solver='full')
        else:
            pca = PCA(n_components=n_components)

        X_train_pca = pca.fit_transform(X_train_scaled)
        print(f"✓ PCA fitted: {pca.n_components_} components")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    else:
        X_train_pca = pca.transform(X_train_scaled)
        print(f"✓ PCA transform applied: {pca.n_components_} components")

    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, pca, pca.explained_variance_ratio_


def prepare_images_for_training(X_train, X_test, use_pca=False, n_components=None,
                                  variance_ratio=0.95, scaler=None, pca=None):
    """Complete preprocessing pipeline for image data: flatten, scale, and optionally apply PCA.

    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Image arrays (N, H, W[, C])
    use_pca : bool
        Whether to apply PCA for dimensionality reduction
    n_components : int or None
        Number of PCA components. If None, uses variance_ratio.
    variance_ratio : float
        Minimum cumulative variance ratio to retain for PCA
    scaler : StandardScaler or None
        Pre-fitted scaler (optional)
    pca : PCA or None
        Pre-fitted PCA transformer (optional)

    Returns:
    --------
    dict with keys:
        - 'X_train_flat': Flattened training data
        - 'X_test_flat': Flattened test data
        - 'X_train_scaled': Scaled training data
        - 'X_test_scaled': Scaled test data
        - 'X_train_final': Final training data (PCA if use_pca, else scaled)
        - 'X_test_final': Final test data (PCA if use_pca, else scaled)
        - 'scaler': Fitted StandardScaler
        - 'pca': Fitted PCA (None if use_pca=False)
        - 'explained_variance': Explained variance ratio (None if use_pca=False)
    """
    # Step 1: Flatten and scale
    X_train_flat, X_test_flat, X_train_scaled, X_test_scaled, scaler = flatten_and_scale_images(
        X_train, X_test, scaler=scaler
    )

    print(f"Flattened shape: {X_train_flat.shape}")
    print(f"Scaled shape: {X_train_scaled.shape}")

    # Step 2: Optional PCA
    if use_pca:
        X_train_pca, X_test_pca, pca, explained_variance = apply_pca(
            X_train_scaled, X_test_scaled,
            n_components=n_components,
            variance_ratio=variance_ratio,
            pca=pca
        )
        X_train_final = X_train_pca
        X_test_final = X_test_pca
        print(f"PCA shape: {X_train_pca.shape}")
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
        pca = None
        explained_variance = None

    return {
        'X_train_flat': X_train_flat,
        'X_test_flat': X_test_flat,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_final': X_train_final,
        'X_test_final': X_test_final,
        'scaler': scaler,
        'pca': pca,
        'explained_variance': explained_variance
    }


def extract_hog_features(images, target_size=(128, 128), pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), visualize=False):
    """Extract HOG (Histogram of Oriented Gradients) features from images.

    This method extracts handcrafted features based on gradient orientations,
    useful for detecting edges and shapes in images.

    Parameters:
    -----------
    images : np.ndarray
        Image array of shape (N, H, W, C) or (N, H, W)
    target_size : tuple
        Target size for resizing images before HOG extraction
    pixels_per_cell : tuple
        Size of a cell for HOG computation
    cells_per_block : tuple
        Number of cells in each block for normalization
    visualize : bool
        If True, also return HOG visualization

    Returns:
    --------
    np.ndarray
        HOG feature matrix of shape (N, D) where D is feature dimension
    """
    features = []

    print(f"Extracting HOG features from {len(images)} images...")

    for idx, image in enumerate(images):
        try:
            # Convert to grayscale if color image
            if image.ndim == 3:
                # Convert RGB to grayscale
                image_gray = np.mean(image, axis=2)
            else:
                image_gray = image

            # Resize image to target size
            img_resized = resize(image_gray, target_size, anti_aliasing=True)

            # Extract HOG features
            hog_features = hog(
                img_resized,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True,
                visualize=False
            )

            # Additional statistical features
            mean_intensity = np.mean(img_resized)
            std_intensity = np.std(img_resized)
            min_intensity = np.min(img_resized)
            max_intensity = np.max(img_resized)

            # Combine HOG features with statistical features
            feature_vector = np.concatenate([
                hog_features,
                [mean_intensity, std_intensity, min_intensity, max_intensity]
            ])

            features.append(feature_vector)

            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(images)} images...")

        except Exception as e:
            print(f"Error extracting HOG features from image {idx}: {e}")
            # Add zero vector as fallback
            features.append(np.zeros(len(features[-1]) if features else 1000))

    feature_array = np.array(features)
    print(f"✓ HOG feature extraction complete: {feature_array.shape}")
    return feature_array


def extract_cnn_features(images, model_name='vgg16', pooling='avg'):
    """Extract deep features using pre-trained CNN models.

    Uses transfer learning from models trained on ImageNet to extract
    high-level features from medical images.

    Parameters:
    -----------
    images : np.ndarray
        Image array of shape (N, H, W, C), values should be in [0, 1]
    model_name : str
        Name of pre-trained model ('vgg16', 'resnet50', 'efficientnetb0')
    pooling : str
        Type of pooling ('avg' or 'max')

    Returns:
    --------
    np.ndarray
        CNN feature matrix of shape (N, D) where D depends on the model
    """
    try:
        # Import tensorflow/keras only when needed
        from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
        import tensorflow as tf

        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')

    except ImportError:
        raise ImportError(
            "TensorFlow is required for CNN feature extraction. "
            "Install it with: pip install tensorflow"
        )

    print(f"Extracting CNN features using {model_name.upper()}...")

    # Select model and preprocessing function
    if model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, pooling=pooling, input_shape=(224, 224, 3))
        preprocess_fn = vgg_preprocess
        feature_dim = 512
    elif model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=(224, 224, 3))
        preprocess_fn = resnet_preprocess
        feature_dim = 2048
    elif model_name.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling=pooling, input_shape=(224, 224, 3))
        preprocess_fn = efficient_preprocess
        feature_dim = 1280
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'vgg16', 'resnet50', or 'efficientnetb0'")

    # Ensure images are in correct format
    images_preprocessed = images.copy()

    # Convert from [0, 1] to [0, 255] if needed
    if images_preprocessed.max() <= 1.0:
        images_preprocessed = images_preprocessed * 255.0

    # Apply model-specific preprocessing
    images_preprocessed = preprocess_fn(images_preprocessed)

    # Extract features
    print(f"  Input shape: {images_preprocessed.shape}")
    features = base_model.predict(images_preprocessed, batch_size=32, verbose=1)

    print(f"✓ CNN feature extraction complete: {features.shape}")
    return features


def extract_features(images, method='hog', **kwargs):
    """Unified interface for feature extraction from images.

    Parameters:
    -----------
    images : np.ndarray
        Image array of shape (N, H, W, C) or (N, H, W)
    method : str
        Feature extraction method:
        - 'hog': Histogram of Oriented Gradients (handcrafted features)
        - 'vgg16': VGG16 CNN features (deep learning)
        - 'resnet50': ResNet50 CNN features (deep learning)
        - 'efficientnetb0': EfficientNetB0 CNN features (deep learning)
        - 'flatten': Just flatten pixels (baseline)
    **kwargs : dict
        Additional arguments passed to specific extraction method

    Returns:
    --------
    np.ndarray
        Feature matrix of shape (N, D)

    Examples:
    ---------
    >>> # HOG features
    >>> X_hog = extract_features(images, method='hog', target_size=(128, 128))

    >>> # VGG16 features
    >>> X_vgg = extract_features(images, method='vgg16', pooling='avg')

    >>> # Baseline (flattened pixels)
    >>> X_flat = extract_features(images, method='flatten')
    """
    print(f"\n{'='*70}")
    print(f"FEATURE EXTRACTION: {method.upper()}")
    print(f"{'='*70}")

    if method.lower() == 'hog':
        return extract_hog_features(images, **kwargs)
    elif method.lower() in ['vgg16', 'resnet50', 'efficientnetb0']:
        return extract_cnn_features(images, model_name=method.lower(), **kwargs)
    elif method.lower() == 'flatten':
        print("Using flattened pixel values (baseline)...")
        return flatten_images(images)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose 'hog', 'vgg16', 'resnet50', 'efficientnetb0', or 'flatten'"
        )


def prepare_images_with_feature_extraction(X_train, X_test, method='hog',
                                           use_pca=False, n_components=None,
                                           variance_ratio=0.95, scaler=None, pca=None,
                                           **feature_kwargs):
    """Complete preprocessing pipeline with feature extraction.

    This is an enhanced version of prepare_images_for_training() that includes
    feature extraction step before scaling and PCA.

    Pipeline: Extract Features → Scale → Optional PCA

    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Image arrays (N, H, W, C)
    method : str
        Feature extraction method ('hog', 'vgg16', 'resnet50', 'efficientnetb0', 'flatten')
    use_pca : bool
        Whether to apply PCA after feature extraction
    n_components : int or None
        Number of PCA components
    variance_ratio : float
        Minimum cumulative variance ratio for PCA
    scaler : StandardScaler or None
        Pre-fitted scaler (optional)
    pca : PCA or None
        Pre-fitted PCA transformer (optional)
    **feature_kwargs : dict
        Additional arguments for feature extraction

    Returns:
    --------
    dict with keys:
        - 'X_train_features': Extracted training features
        - 'X_test_features': Extracted test features
        - 'X_train_scaled': Scaled training features
        - 'X_test_scaled': Scaled test features
        - 'X_train_final': Final training data (PCA if use_pca, else scaled)
        - 'X_test_final': Final test data (PCA if use_pca, else scaled)
        - 'scaler': Fitted StandardScaler
        - 'pca': Fitted PCA (None if use_pca=False)
        - 'explained_variance': Explained variance ratio (None if use_pca=False)
        - 'method': Feature extraction method used
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING PIPELINE WITH FEATURE EXTRACTION")
    print(f"{'='*70}")

    # Step 1: Extract features
    X_train_features = extract_features(X_train, method=method, **feature_kwargs)
    X_test_features = extract_features(X_test, method=method, **feature_kwargs)

    print(f"\nFeature extraction complete:")
    print(f"  Training features: {X_train_features.shape}")
    print(f"  Test features: {X_test_features.shape}")

    # Step 2: Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
    else:
        X_train_scaled = scaler.transform(X_train_features)

    X_test_scaled = scaler.transform(X_test_features)
    print(f"✓ Features scaled with StandardScaler")

    # Step 3: Optional PCA
    if use_pca:
        print(f"\nApplying PCA (variance_ratio={variance_ratio})...")
        X_train_pca, X_test_pca, pca, explained_variance = apply_pca(
            X_train_scaled, X_test_scaled,
            n_components=n_components,
            variance_ratio=variance_ratio,
            pca=pca
        )
        X_train_final = X_train_pca
        X_test_final = X_test_pca
        print(f"  Final shape after PCA: {X_train_final.shape}")
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
        pca = None
        explained_variance = None
        print(f"  No PCA applied, using scaled features")

    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Final training data: {X_train_final.shape}")
    print(f"Final test data: {X_test_final.shape}")

    return {
        'X_train_features': X_train_features,
        'X_test_features': X_test_features,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_final': X_train_final,
        'X_test_final': X_test_final,
        'scaler': scaler,
        'pca': pca,
        'explained_variance': explained_variance,
        'method': method
    }


if __name__ == "__main__":
    # Test the module
    print("Preprocessing module test")
    print(f"DX_DICT: {DX_DICT}")
    print(f"Malignant types: {MALIGNANT_TYPES}")
    print(f"Benign types: {BENIGN_TYPES}")