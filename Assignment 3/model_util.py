"""
Utility functions for model training, evaluation, and analysis.

This module provides helper functions for training and evaluating neural networks
in incremental learning experiments. It includes functions for model evaluation,
overfitting detection, and training analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Any
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 0) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model: torch.nn.Module, features: np.ndarray, labels: np.ndarray, 
                   active_classes: List[int]) -> Dict[str, Any]:
    """
    Evaluate model performance on given features and labels.
    """
    # Handle empty dataset
    if len(features) == 0:
        return {
            'accuracy': 0, 
            'f1_score': 0, 
            'predictions': [], 
            'cross_entropy': float('inf'),
            'mcc': 0
        }

    # Create mappings between class labels and model output indices
    class_to_index = {cls: idx for idx, cls in enumerate(active_classes)}
    index_to_class = {idx: cls for idx, cls in enumerate(active_classes)}
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Convert features to tensor and get model predictions
        features_tensor = torch.from_numpy(features.astype(np.float32))
        outputs = model(features_tensor)
        probs = F.softmax(outputs, dim=1)
        
        # Handle case where model has no output classes yet
        if outputs.shape[1] == 0:
            predictions = np.array([-1] * len(features))
            cross_entropy = float('inf')
            mcc = 0
        else:
            # Get predicted class indices and convert back to original labels
            _, predicted_indices = torch.max(outputs, 1)
            predicted_indices = predicted_indices.numpy()
            predictions = np.array([index_to_class.get(idx, -1) for idx in predicted_indices]).flatten()
            
            # Calculate cross-entropy loss
            mapped_labels = torch.tensor([class_to_index.get(int(label), 0) for label in labels])
            cross_entropy = F.cross_entropy(outputs, mapped_labels).item()
            
    # Calculate metrics only for valid predictions
    valid_mask = predictions != -1

    if np.sum(valid_mask) == 0:
        return {
            'accuracy': 0, 
            'f1_score': 0, 
            'predictions': predictions, 
            'cross_entropy': cross_entropy,
            'mcc': mcc
        }
    
    # Extract valid predictions and labels for metric calculation
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    probs = probs[valid_mask]
    
    # Calculate performance metrics
    accuracy = balanced_accuracy_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions, average='macro', zero_division=0)
    mcc = matthews_corrcoef(valid_labels, valid_predictions)
    
    return {
        'accuracy': accuracy, 
        'f1_score': f1, 
        'predictions': predictions, 
        'cross_entropy': cross_entropy,
        'mcc': mcc
    }


def detect_stagnation(loss_history: List[float], window: int = 5, tolerance: float = 0.001) -> bool:
    """
    Detect if training has stagnated based on loss history.
    """
    if len(loss_history) < window + 1:
        return False
    
    # Get recent loss values
    recent_losses = loss_history[-window:]
    
    # Calculate changes between consecutive losses
    changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
    avg_change = np.mean(changes)
    
    # Training is stagnated if average change is below tolerance
    return avg_change < tolerance


def is_increasing(values: List[float], window: int = 3, threshold: float = 1e-5) -> bool:
    """
    Check if values are consistently increasing over a window.
    """
    if len(values) < window:
        return False
    
    # Check if all differences in the window are above threshold
    return np.all(np.diff(values[-window:]) > threshold)


def fit_check(train_losses: List[float], val_losses: List[float], val_accuracies: List[float], 
              num_classes: int, patience_window: int = 10) -> Tuple[bool, bool]:
    """
    Detect underfitting and overfitting based on loss patterns and accuracy trends.
    """
    if len(train_losses) < patience_window or len(val_losses) < patience_window:
        return False, False
    
    # Detect stagnation in training and validation losses
    train_stag = detect_stagnation(train_losses, window=patience_window, tolerance=0.001)
    val_stag = detect_stagnation(val_losses, window=patience_window, tolerance=0.001)
    
    # Detect underfitting and overfitting
    underfit = detect_underfitting(train_losses, val_losses, train_stag, val_stag, num_classes)
    overfit = detect_overfitting(train_losses, val_losses, train_stag, val_stag, patience_window=patience_window)
    
    return underfit, overfit


def detect_underfitting(train_losses: List[float], val_losses: List[float], 
                        train_stag: bool, val_stag: bool, num_classes: int) -> bool:
    """
    Detect underfitting based on loss patterns.
    """
    # High loss threshold: should be significantly above random guessing
    base_loss = np.log(num_classes)
    high_loss_threshold = base_loss * 0.6  # 60% of random guessing loss
    
    # Check if losses haven't improved much from initial values
    initial_train_loss = train_losses[0] if len(train_losses) > 0 else float('inf')
    initial_val_loss = val_losses[0] if len(val_losses) > 0 else float('inf')
    improvement_threshold = 0.1  # Minimum improvement required
    
    # Underfitting if
    # Both losses stagnate at high values, or
    # Losses haven't improved much from initial values
    underfit = ((train_stag and val_stag and 
                val_losses[-1] > high_loss_threshold and 
                train_losses[-1] > high_loss_threshold) or
               (train_stag and val_stag and
                (initial_train_loss - train_losses[-1]) < improvement_threshold and
                (initial_val_loss - val_losses[-1]) < improvement_threshold))
    
    return underfit


def detect_overfitting(train_losses: List[float], val_losses: List[float], 
                       train_stag: bool, val_stag: bool, patience_window: int) -> bool:
    """
    Detect overfitting based on loss patterns.
    """
    # Overfitting: validation loss starts increasing while train loss decreases
    val_increasing = is_increasing(val_losses, window=patience_window)
    train_decreasing = not detect_stagnation(train_losses, window=patience_window, tolerance=0.001)
    
    # Gap-based overfitting detection
    loss_gap = val_losses[-1] - train_losses[-1]
    gap_threshold = 0.1  # Significant gap threshold
    
    # Simplified overfitting detection (less aggressive)
    overfit = (val_increasing or 
               (train_decreasing and val_stag and loss_gap > gap_threshold))
    
    return overfit