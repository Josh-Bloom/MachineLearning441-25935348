"""
Dataset loading utilities.

Datasets:
  - Iris (3-class)
  - Wine Quality (red/white)
  - Letter Recognition

References:
  - https://archive.ics.uci.edu/dataset/53/iris
  - https://archive.ics.uci.edu/dataset/186/wine+quality
  - https://archive.ics.uci.edu/dataset/59/letter+recognition
"""

import pandas as pd
import numpy as np
from typing import Tuple
from config import SEED

def load_iris() -> pd.DataFrame:
    """Load and preprocess the Iris dataset.
    
    Returns:
        DataFrame with 4 features and class labels (0-2).
    """
    # Load raw data (no header in original file)
    iris = pd.read_csv('data/iris/iris.data', header=None)
    
    # Assign meaningful column names
    iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    # Convert string class labels to integers for sklearn compatibility
    iris['class'] = iris['class'].map({
        'Iris-setosa': 0, 
        'Iris-versicolor': 1, 
        'Iris-virginica': 2
    })
    
    return iris


def load_wine_red() -> pd.DataFrame:
    """
    Load and preprocess the red wine quality dataset.
    
    Returns:
        DataFrame with 11 features and quality labels (0-5).
    """
    # Load CSV with semicolon delimiter
    wine_red = pd.read_csv('data/wine/winequality-red.csv', sep=';')
    
    # Standardise column names (replace spaces with underscores)
    wine_red.columns = wine_red.columns.str.replace(' ', '_')
    
    # Map quality scores (3-8) to zero-indexed classes (0-5)
    wine_red['quality'] = wine_red['quality'].map({3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5})
    
    return wine_red


def load_wine_white() -> pd.DataFrame:
    """
    Load and preprocess the white wine quality dataset.
    
    Returns:
        DataFrame with 11 features and quality labels (0-6).
    """
    # Load CSV with semicolon delimiter
    wine_white = pd.read_csv('data/wine/winequality-white.csv', sep=';')
    
    # Standardise column names (replace spaces with underscores)
    wine_white.columns = wine_white.columns.str.replace(' ', '_')
    
    # Map quality scores (3-9) to zero-indexed classes (0-6)
    wine_white['quality'] = wine_white['quality'].map({
        3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6
    })
    
    return wine_white


def load_letter_recognition() -> pd.DataFrame:
    """
    Load and preprocess the letter recognition dataset.
    
    Returns:
        DataFrame with 16 features and letter labels (0-25 for A-Z).
    """
    # Load raw data (no header in original file)
    letter_recognition = pd.read_csv('data/letter/letter-recognition.data', header=None)
    
    # First column contains letters 'A'-'Z'; convert to integers 0-25
    # ord('A') = 65, so ord(letter) - ord('A') gives 0-25
    letter_recognition.iloc[:, 0] = letter_recognition.iloc[:, 0].map(
        lambda x: ord(x) - ord('A')
    ).astype(int)
    
    return letter_recognition

def dataset_loader(
    dataset_name: str, 
    test_size: float = 0.2, 
    seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset and split into train/test sets.
    """
    # Load the appropriate dataset based on name
    if dataset_name == 'iris':
        dataset = load_iris()
        X = dataset.drop(columns=['class'])
        y = dataset['class']
        
    # Optionally enable red wine dataset
    # elif dataset_name == 'wine_red':
    #     dataset = load_wine_red()
    #     X = dataset.drop(columns=['quality'])
    #     y = dataset['quality']
    
    elif dataset_name == 'wine':
        dataset = load_wine_white()
        X = dataset.drop(columns=['quality'])
        y = dataset['quality']
        
    elif dataset_name == 'letter':
        dataset = load_letter_recognition()
        # First column is the target, rest are features
        X = dataset.iloc[:, 1:]
        y = dataset.iloc[:, 0].astype(int)
    
    # Perform stratified train/test split
    # Stratification maintains class proportions in both sets
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, 
                           random_state=seed, stratify=y)

if __name__ == '__main__':
    """
    Test all dataset loaders and print dataset statistics.
    """
    import numpy as np
    
    # List of datasets to test
    datasets = ['iris', 'wine', 'letter']
    
    for dataset_name in datasets:
        try:
            # Load dataset with default 80/20 train/test split
            X_train, X_test, y_train, y_test = dataset_loader(dataset_name)
            
            # Compute statistics
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]
            n_total = n_train + n_test
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            
            # Class distribution in training set
            unique, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip(unique, counts))
            
            # Print summary
            print(f"\n{dataset_name.upper()}")
            print(f"  Total samples: {n_total}")
            print(f"  Train/Test split: {n_train}/{n_test} ({n_train/n_total*100:.1f}%/{n_test/n_total*100:.1f}%)")
            print(f"  Features: {n_features}")
            print(f"  Classes: {n_classes}")
            print(f"  Class distribution (train): {class_dist}")
            
        except Exception as e:
            print(f"\n{dataset_name.upper()}: ERROR - {e}")
