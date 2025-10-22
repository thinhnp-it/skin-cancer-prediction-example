"""
Image Data Loading Module
Handles loading and preprocessing of HAM10000 skin lesion images
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_image_path(image_id, base_path='../data'):
    """
    Find the full path to an image file given its image_id

    Parameters:
    -----------
    image_id : str
        Image ID (e.g., 'ISIC_0024306')
    base_path : str
        Base path to data directory

    Returns:
    --------
    str or None
        Full path to image file, or None if not found
    """
    # Check in both image directories
    part1_path = os.path.join(base_path, 'HAM10000_images_part_1', f'{image_id}.jpg')
    part2_path = os.path.join(base_path, 'HAM10000_images_part_2', f'{image_id}.jpg')

    if os.path.exists(part1_path):
        return part1_path
    elif os.path.exists(part2_path):
        return part2_path
    else:
        return None


def load_and_preprocess_image(image_path, target_size=(224, 224), normalize=True):
    """
    Load and preprocess a single image

    Parameters:
    -----------
    image_path : str
        Path to image file
    target_size : tuple
        Target size (height, width) for resizing
    normalize : bool
        Whether to normalize pixel values to [0, 1]

    Returns:
    --------
    np.ndarray
        Preprocessed image array of shape (height, width, channels)
    """
    try:
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if necessary (some images might be grayscale or RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize pixel values to [0, 1] if requested
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0

        return img_array

    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None


def load_images_from_metadata(metadata_df, base_path='../data', target_size=(224, 224),
                               normalize=True, verbose=True):
    """
    Load all images referenced in metadata DataFrame

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        DataFrame containing 'image_id' column
    base_path : str
        Base path to data directory
    target_size : tuple
        Target size (height, width) for resizing
    normalize : bool
        Whether to normalize pixel values to [0, 1]
    verbose : bool
        Whether to print progress

    Returns:
    --------
    np.ndarray, list
        Array of images (N, H, W, C) and list of successfully loaded image_ids
    """
    images = []
    loaded_image_ids = []
    failed_count = 0

    total_images = len(metadata_df)

    for idx, row in metadata_df.iterrows():
        image_id = row['image_id']

        # Find image path
        img_path = get_image_path(image_id, base_path)

        if img_path is None:
            if verbose and failed_count < 5:  # Only print first 5 failures
                print(f"Warning: Image not found for {image_id}")
            failed_count += 1
            continue

        # Load and preprocess image
        img_array = load_and_preprocess_image(img_path, target_size, normalize)

        if img_array is not None:
            images.append(img_array)
            loaded_image_ids.append(image_id)
        else:
            failed_count += 1

        # Print progress
        if verbose and (idx + 1) % 1000 == 0:
            print(f"Loaded {idx + 1}/{total_images} images...")

    if verbose:
        print(f"\nSuccessfully loaded: {len(images)}/{total_images} images")
        if failed_count > 0:
            print(f"Failed to load: {failed_count} images")

    # Convert to numpy array
    images_array = np.array(images)

    return images_array, loaded_image_ids


def create_image_augmentation_pipeline():
    """
    Create an image augmentation pipeline for training data
    This is a placeholder - actual implementation would use libraries like
    albumentations or imgaug for production use

    Returns:
    --------
    dict
        Dictionary with augmentation parameters
    """
    augmentation_config = {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation_range': 20,  # degrees
        'zoom_range': 0.1,
        'brightness_range': (0.8, 1.2),
        'fill_mode': 'reflect'
    }

    return augmentation_config


def verify_image_metadata_match(metadata_df, base_path='../data'):
    """
    Verify that all images in metadata exist in the image directories

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        DataFrame containing 'image_id' column
    base_path : str
        Base path to data directory

    Returns:
    --------
    dict
        Dictionary with verification statistics
    """
    total_images = len(metadata_df['image_id'].unique())
    found_count = 0
    missing_images = []

    for image_id in metadata_df['image_id'].unique():
        img_path = get_image_path(image_id, base_path)
        if img_path is not None:
            found_count += 1
        else:
            missing_images.append(image_id)

    stats = {
        'total_unique_images': total_images,
        'found': found_count,
        'missing': len(missing_images),
        'missing_image_ids': missing_images[:10]  # Only first 10
    }

    print(f"\nImage Verification:")
    print(f"Total unique images in metadata: {stats['total_unique_images']}")
    print(f"Found: {stats['found']}")
    print(f"Missing: {stats['missing']}")

    if stats['missing'] > 0:
        print(f"\nFirst 10 missing image IDs: {stats['missing_image_ids']}")

    return stats


def batch_image_generator(image_ids, metadata_df, base_path='../data',
                          target_size=(224, 224), batch_size=32, normalize=True):
    """
    Generator that yields batches of images and labels
    Useful for loading large datasets that don't fit in memory

    Parameters:
    -----------
    image_ids : list
        List of image IDs to load
    metadata_df : pd.DataFrame
        DataFrame with image_id and label columns
    base_path : str
        Base path to data directory
    target_size : tuple
        Target image size
    batch_size : int
        Number of images per batch
    normalize : bool
        Whether to normalize images

    Yields:
    -------
    tuple
        (batch_images, batch_labels) arrays
    """
    num_samples = len(image_ids)

    while True:  # Infinite generator for training
        # Shuffle at start of each epoch
        indices = np.random.permutation(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                image_id = image_ids[idx]
                img_path = get_image_path(image_id, base_path)

                if img_path is not None:
                    img_array = load_and_preprocess_image(img_path, target_size, normalize)

                    if img_array is not None:
                        batch_images.append(img_array)
                        # Get label from metadata
                        label = metadata_df[metadata_df['image_id'] == image_id].iloc[0]['binary_label']
                        batch_labels.append(label)

            if len(batch_images) > 0:
                yield np.array(batch_images), np.array(batch_labels)


if __name__ == "__main__":
    # Test the module
    print("Image Loader Module Test")
    print("=" * 60)

    # Test loading metadata
    metadata_path = 'data/HAM10000_metadata.csv'
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        print(f"\nMetadata loaded: {len(df)} samples")

        # Verify images
        stats = verify_image_metadata_match(df)

        # Test loading a single image
        sample_image_id = df.iloc[0]['image_id']
        print(f"\nTesting with sample image: {sample_image_id}")

        img_path = get_image_path(sample_image_id)
        if img_path:
            print(f"Image path: {img_path}")

            # Load and preprocess
            img_array = load_and_preprocess_image(img_path, target_size=(224, 224))
            if img_array is not None:
                print(f"Loaded image shape: {img_array.shape}")
                print(f"Image dtype: {img_array.dtype}")
                print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        else:
            print("Image not found!")
    else:
        print(f"Metadata file not found at {metadata_path}")