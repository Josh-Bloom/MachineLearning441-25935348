"""
Data loading and preprocessing utilities for machine learning experiments.

This module provides functions for loading and preprocessing datasets from the UCI ML Repository.
It handles data standardisation, class imbalance, and train/test splitting for the incremental
learning experiments.

Datasets:
- glass: Glass identification (6 classes)
- sil: Vehicle silhouettes (4 classes) 
- segment: Image segmentation (7 classes)
- wine: Wine quality (6 classes)
- yeast: Yeast protein localisation (10 classes)
"""

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple
import logging
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Dataset ID mapping for UCI ML Repo
DATASET_IDS = {
    "glass": 42, # https://archive.ics.uci.edu/dataset/42/glass+identification
    "sil": 149, # https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
    "segment": 50, # https://archive.ics.uci.edu/dataset/50/image+segmentation
    "wine": 186, # https://archive.ics.uci.edu/dataset/186/wine+quality
    "yeast": 110, # https://archive.ics.uci.edu/dataset/110/yeast
}

# Downsampling parameters for class imbalance handling
# Negative values create more balanced class distributions
DOWNSAMPLE_ALPHAS = {
    "sil": -0.2,      # Moderate downsampling for vehicle dataset
    "segment": -0.3,  # More aggressive downsampling for segmentation dataset
}

def fetch_data(dataset_name: str, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch and preprocess a dataset from UCI ML Repository.
    """
    # Validate dataset name
    if dataset_name not in DATASET_IDS:
        available_datasets = list(DATASET_IDS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {available_datasets}")
    
    try:
        logger.info(f"Fetching dataset: {dataset_name}")

        # Load dataset from cache or UCI repository
        dataset = fetch_dataset(dataset_name)
        
        # Extract features and labels
        X = dataset.data.features
        y = dataset.data.targets

        # Handle specific dataset issues
        if dataset_name == 'sil':  # Remove missing value from vehicle dataset
            X = X.drop(index=[752])
            y = y.drop(index=[752])

        logger.info(f"Dataset loaded successfully. Shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Preprocess the data (standardisation, downsampling, train/test split)
        X_train, y_train, X_test, y_test = _preprocess_data(
            X, y,
            downsample_alpha=DOWNSAMPLE_ALPHAS.get(dataset_name, None),
            test_size=test_size, 
            seed=seed
        )
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Failed to fetch dataset '{dataset_name}': {str(e)}")
        raise Exception(f"Dataset loading failed: {str(e)}")


def _preprocess_data(X, y, downsample_alpha=None, test_size=0.2, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the dataset by standardising features and handling class imbalance.
    """
    try:
        # Step 1: Standardise features (mean=0, std=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 2: Convert labels to numpy array and ensure proper shape
        y_array = np.array(y).flatten()

        # Convert string labels to integer indices if needed
        if y_array.dtype == 'object':
            unique_labels, y_array = np.unique(y_array, return_inverse=True)

        # Step 3: Validate the preprocessed data
        _validate_data(X_scaled, y_array)

        # Step 4: Induce class imbalance with downsampling (if specified)
        if downsample_alpha is not None:
            print(f"Imbalance ratio before downsampling: {get_imbalance_ratio(y_array):.3f}")
            X_scaled, y_array, X_remain, y_remain = downsample(X_scaled, y_array, downsample_alpha, seed=seed)
            print(f"Imbalance ratio after downsampling: {get_imbalance_ratio(y_array):.3f}")
        else:
            print(f"Imbalance ratio: {get_imbalance_ratio(y_array):.3f}")
            
        # Step 5: Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_array,
            test_size=test_size, 
            random_state=seed,
            stratify=y_array,  # Maintain class proportions
            shuffle=True
        )
        
        # Step 6: Add removed samples to test set (for downsampling)
        if downsample_alpha is not None:
            X_test = np.concatenate((X_test, X_remain), axis=0)
            y_test = np.concatenate((y_test, y_remain), axis=0)
        
        logger.info(f"Data preprocessed successfully. Features standardised, labels converted.")

        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise Exception(f"Data preprocessing failed: {str(e)}")


def _validate_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the preprocessed data for common issues.
    """
    # Check for empty data
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty dataset detected")

    # Check for NaN values in features
    if DataFrame(X).isna().sum().sum() > 0:
        raise ValueError("NaN values detected in features")
    
    # Check for NaN values in labels
    if Series(y).isna().sum() > 0:
        raise ValueError("NaN values detected in labels")
    
    # Check data shapes (number of samples must match)
    if len(X) != len(y):
        raise ValueError(f"Feature and label count mismatch: {len(X)} vs {len(y)}")
    
    # Check for sufficient classes (need at least 2 for classification)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Insufficient classes for classification: {len(unique_classes)}")

def downsample(X: np.ndarray, y: np.ndarray, alpha: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample dataset to handle class imbalance using a tail-ratio distribution.
    """
    from imblearn.under_sampling import RandomUnderSampler
    
    # Get class information
    num_classes = len(np.unique(y))
    counts = Series(y).value_counts()
    class_labels = counts.index
    counts = counts.to_numpy()
    top_freq = counts[0]  # Frequency of most common class
    
    # Calculate tail-ratio distribution relative to the majority class
    # Negative alpha values create more balanced distributions
    ratios = np.arange(1, num_classes + 1) ** (-1 * (1 + 1.0/(num_classes) + alpha))
    
    # Construct sampling strategy for each class
    sampling_strategy = {}
    for cls, ratio, org_count in zip(class_labels, ratios, counts):
        # Cannot sample more than what exists
        sampling_strategy[cls] = min(int(round(ratio * top_freq)), org_count)
    
    # Use RandomUnderSampler to perform the downsampling
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)

    # Get indices of samples that were kept
    kept_idx = rus.sample_indices_

    # Find indices of samples that were removed
    all_idx = np.arange(len(y))
    removed_idx = np.setdiff1d(all_idx, kept_idx)

    # Extract removed samples
    X_removed, y_removed = X[removed_idx], y[removed_idx]

    return X_res, y_res, X_removed, y_removed

def get_imbalance_ratio(y: np.ndarray) -> float:
    """
    Calculate the imbalance ratio of a dataset as the coefficient of variation of class counts.
    """
    # Get class counts
    counts = np.unique(y, return_counts=True)[1]
    
    # Calculate coefficient of variation (std/mean) as imbalance measure
    # This is more robust than max/min ratio for multi-class problems
    return np.std(counts) / np.mean(counts)

def fetch_dataset(name: str):
    """
    Fetch a dataset from UCI ML Repository or from local cache.
    """
    import os
    import joblib

    # Determine the directory containing this file
    try:
        # Normal script: use __file__
        file_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        # Jupyter notebook or interactive session: use current working directory
        file_dir = os.getcwd()

    # Check if dataset is cached locally
    cache_path = f"{file_dir}/datasets/{name}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading dataset from cache: {name}")
        return joblib.load(cache_path)
    else:
        # Download from UCI ML Repository
        print(f"Fetching dataset from UCI ML Repository: {expand_name(name)}")
        return fetch_ucirepo(id=DATASET_IDS[name])

def expand_name(name: str) -> str:
    """
    Expand dataset name to full descriptive name.
    """
    name_mapping = {
        "glass": "Glass Identification",
        "sil": "Vehicle Silhouettes", 
        "segment": "Image Segmentation",
        "wine": "Wine Quality",
        "yeast": "Yeast",
    }
    
    if name not in name_mapping:
        raise KeyError(f"Unknown dataset name: {name}")
    
    return name_mapping[name]