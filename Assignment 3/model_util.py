import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# Set random seeds for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_model(model, features, labels, active_classes):
    """Evaluate model performance"""
    if len(features) == 0:
        return {'accuracy': 0, 'f1_score': 0, 'predictions': [], 'cross_entropy': float('inf')}

    class_to_index = {cls: idx for idx, cls in enumerate(active_classes)}
    index_to_class = {idx: cls for idx, cls in enumerate(active_classes)}
    
    model.eval()
    with torch.no_grad():
        features_tensor = torch.from_numpy(features.astype(np.float32))
        outputs = model(features_tensor)
        
        if outputs.shape[1] == 0:
            predictions = np.array([-1] * len(features))
            cross_entropy = float('inf')
        else:
            _, predicted_indices = torch.max(outputs, 1)
            predicted_indices = predicted_indices.numpy()
            predictions = np.array([index_to_class.get(idx, -1) for idx in predicted_indices]).flatten()
            # print(predictions.flatten())
            # Calculate cross-entropy loss for all samples
            # Map labels to indices for loss calculation
            mapped_labels = torch.tensor([class_to_index.get(int(label[0]), 0) for label in labels])
            cross_entropy = F.cross_entropy(outputs, mapped_labels).item()
    
    # Calculate accuracy and F1 score only for valid predictions
    valid_mask = predictions != -1

    if np.sum(valid_mask) == 0:
        return {'accuracy': 0, 'f1_score': 0, 'predictions': predictions, 'cross_entropy': cross_entropy}
    
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    

    accuracy = balanced_accuracy_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions, average='macro', zero_division=0)
    
    return {'accuracy': accuracy, 'f1_score': f1, 'predictions': predictions, 'cross_entropy': cross_entropy}



def detect_stagnation(loss_history, window=5, tolerance=0.001):
    """Detect if training has stagnated"""
    if len(loss_history) < window + 1:
        return False
    
    recent_losses = loss_history[-window:]
    changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
    avg_change = np.mean(changes)
    
    return avg_change < tolerance

def is_increasing(losses, window=3, threshold=1e-5):
    if len(losses) < window:
        return False
    return np.all(np.diff(losses[-window:]) > threshold)

def fit_check(train_losses, val_losses, val_accuracies, num_classes, patience_window=10):
    """Detect underfitting and overfitting based on loss patterns and accuracy trends"""
    if len(train_losses) < patience_window or len(val_losses) < patience_window:
        return False, False
    
    train_stag = detect_stagnation(train_losses, window=patience_window, tolerance=0.001)
    val_stag = detect_stagnation(val_losses, window=patience_window, tolerance=0.001)
    
    # Underfitting: both train and val losses stagnate at high values
    # High loss threshold: should be significantly above random guessing
    random_guess_loss = np.log(num_classes)
    high_loss_threshold = random_guess_loss * 0.6  # 60% of random guessing loss (more sensitive)
    
    # Also check if losses are not improving much from initial values
    initial_train_loss = train_losses[0] if len(train_losses) > 0 else float('inf')
    initial_val_loss = val_losses[0] if len(val_losses) > 0 else float('inf')
    
    # Underfitting if:
    # 1. Both losses stagnate at high values, OR
    # 2. Losses haven't improved much from initial values
    improvement_threshold = 0.1  # Minimum improvement required
    
    underfit = ((train_stag and val_stag and 
                val_losses[-1] > high_loss_threshold and 
                train_losses[-1] > high_loss_threshold) or
               (train_stag and val_stag and
                (initial_train_loss - train_losses[-1]) < improvement_threshold and
                (initial_val_loss - val_losses[-1]) < improvement_threshold))
    
    # Overfitting: validation loss starts increasing while train loss decreases
    # or validation loss is significantly higher than training loss
    val_increasing = is_increasing(val_losses, window=patience_window//2)
    train_decreasing = not detect_stagnation(train_losses, window=patience_window//2, tolerance=0.001)
    
    # Gap-based overfitting detection
    loss_gap = val_losses[-1] - train_losses[-1]
    gap_threshold = 0.1  # Significant gap threshold
    
    # Simplified overfitting detection (less aggressive)
    overfit = (val_increasing or 
               (train_decreasing and val_stag and loss_gap > gap_threshold))
    
    return underfit, overfit