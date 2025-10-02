"""
Core training and evaluation functions for incremental learning experiments.

Implements training loops and comparison functions for incremental learning using
dynamic neural networks that can grow their architecture during training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import copy
from typing import Dict, List, Tuple, Optional, Any
import logging

from model_util import (
    set_seed, 
    evaluate_model, 
    fit_check
)
from dynamicNN import DynamicNetwork
from data import fetch_data
from config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _create_optimiser_and_scheduler(model: DynamicNetwork, config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Create optimiser and learning rate scheduler for training.
    """
    # Get learning parameters from config
    learning_rate = config.get('learning_rate', 0.001)
    patience_window = config.get('patience', 10)
    weight_decay = config.get('weight_decay', 1e-5)
    
    # Create Adam optimiser with weight decay for regularisation
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 
        mode='min',           # Reduce when validation loss stops decreasing
        factor=0.8,           # Reduce learning rate by 20%
        patience=patience_window,  # Wait this many epochs before reducing
        min_lr=learning_rate*0.01  # Don't reduce below 1% of original LR
    )
    
    return optimiser, scheduler


def _train_single_epoch(model: DynamicNetwork, train_loader: DataLoader, optimiser: torch.optim.Optimizer, 
                       class_to_index: Dict[int, int]) -> float:
    """
    Train the model for a single epoch using mini-batch gradient descent.
    """
    model.train()  # Set model to training mode
    epoch_losses = []
    
    # Process each batch in the training data
    for batch_features, batch_labels in train_loader:
        # Clear gradients from previous batch
        optimiser.zero_grad()
        
        # Forward pass: get model predictions
        outputs = model(batch_features)
        
        # Map class labels to model output indices (0, 1, 2, ...)
        mapped_labels = torch.tensor([class_to_index[int(label)] for label in batch_labels])
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(outputs, mapped_labels)
        
        # Backward pass
        loss.backward()
        optimiser.step()
        
        # Store loss for this batch
        epoch_losses.append(loss.item())
    
    # Return average loss across all batches
    return np.mean(epoch_losses)


def _evaluate_and_update_history(model: DynamicNetwork, X_val: np.ndarray, y_val: np.ndarray, 
                                current_classes: List[int], training_history: Dict[str, List]) -> Dict[str, float]:
    """
    Evaluate model on validation set and update training history.
    """
    # Evaluate model performance on validation set
    val_results = evaluate_model(model, X_val, y_val, current_classes)
    
    # Extract key metrics
    val_accuracy = val_results.get('accuracy', 0)
    val_f1 = val_results.get('f1_score', 0)
    val_loss = val_results.get('cross_entropy', float('inf'))
    val_mcc = val_results.get('mcc', 0)
    
    # Update training history with current validation metrics
    training_history['val_accuracy'].append(val_accuracy)
    training_history['val_loss'].append(val_loss)
    training_history['val_f1'].append(val_f1)
    training_history['val_mcc'].append(val_mcc)
    
    return val_results


def _train_single_architecture(model: DynamicNetwork, train_loader: DataLoader, val_data: Tuple[np.ndarray, np.ndarray],
                             current_classes: List[int], config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Train a single architecture until convergence or overfitting.
    """
    X_val, y_val = val_data
    
    # Set up optimiser and learning rate scheduler
    optimiser, scheduler = _create_optimiser_and_scheduler(model, config)
    
    # Initialise training history tracking
    training_history = {
        'train_loss': [], 'val_accuracy': [], 'val_f1': [], 
        'val_loss': [], 'val_mcc': []
    }
    
    # Track best model state for restoration
    best_val_loss = float('inf')
    best_history = None
    best_model_state = None
    
    # Get training parameters
    max_epochs = config.get('max_epochs', 100)
    patience_window = config.get('patience', 10)
    class_to_index = {cls: idx for idx, cls in enumerate(current_classes)}
    
    # Training loop
    for epoch in range(max_epochs):
        # Train for one epoch
        avg_epoch_loss = _train_single_epoch(model, train_loader, optimiser, class_to_index)
        training_history['train_loss'].append(avg_epoch_loss)
        
        # Evaluate on validation set
        val_results = _evaluate_and_update_history(model, X_val, y_val, current_classes, training_history)
        val_loss = val_results.get('cross_entropy', float('inf'))
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Store best model state if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_history = copy.deepcopy(training_history)
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Check for overfitting after sufficient training epochs
        if epoch >= patience_window:
            _, current_overfit = fit_check(
                training_history['train_loss'], 
                training_history['val_loss'], 
                training_history['val_accuracy'],
                num_classes=len(current_classes),
                patience_window=patience_window
            )
            
            if current_overfit:
                logger.info(f"Overfitting detected at epoch {epoch}! Stopping training.")
                break
    
    # Restore to best model state found during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_history or training_history, current_overfit


def train_model_with_architecture_search(model: DynamicNetwork, train_loader: DataLoader, 
                                        val_data: Tuple[np.ndarray, np.ndarray], 
                                        current_classes: List[int], config: Dict[str, Any], 
                                        stage_info: Optional[Dict] = None, 
                                        history: Optional[Dict] = None) -> Tuple[DynamicNetwork, bool, bool]:
    """
    Train model with incremental architecture growth based on overfitting detection.
    """
    X_val, y_val = val_data
    
    # Initialise architecture search state
    hidden_neurons_added = 0
    max_hidden_neurons = config.get('max_hidden_neurons', 20)
    overfit = False
    
    logger.info(f"Starting architecture search with max {max_hidden_neurons} hidden neurons")
    
    # Main architecture search loop
    while hidden_neurons_added < max_hidden_neurons:
        if config.get('verbose', True):
            logger.info(f"Training with {model.hidden_neurons} hidden neurons...")
        
        # Train the current architecture until convergence or overfitting
        best_history, current_overfit = _train_single_architecture(
            model, train_loader, val_data, current_classes, config
        )
        
        # Store checkpoint for this architecture (for potential restoration)
        if best_history is not None:
            val_f1 = best_history['val_f1'][-1] if best_history['val_f1'] else 0
            model.store_architecture_checkpoint(val_f1, model.hidden_neurons)
        
        # Update training history for logging/analysis
        if history is not None and stage_info is not None and best_history is not None:
            _update_training_history(history, stage_info, current_classes, model, best_history, current_overfit)
        
        # Log current architecture performance
        if config.get('verbose', True):
            logger.info(f"Best validation accuracy: {best_history['val_accuracy'][-1]:.4f}")
            logger.info(f"Best validation loss (cross entropy): {best_history['val_loss'][-1]:.4f}")
            logger.info(f"Best validation MCC: {best_history['val_mcc'][-1]:.4f}")
        
        # If overfitting detected, stop architecture growth
        if current_overfit:
            overfit = True
            logger.info("Overfitting detected! Stopping architecture growth.")
            break
        
        # Assume underfitting - add hidden neuron and continue
        model.add_hidden_neuron()
        hidden_neurons_added += 1
        
        # Adjust training parameters for fine-tuning (when preserving weights)
        if config.get('model_mode', 'reinit') == 'preserve':
            config['max_epochs'] = config.get('fine_max_epochs', 30)
            config['learning_rate'] = config.get('fine_learning_rate', 0.0005)
            config['patience'] = config.get('fine_patience', 5)
    
    # Restore to best architecture found during search
    if model.restore_best_architecture():
        if config.get('verbose', True):
            logger.info(f"Restored to best architecture: F1={model.best_architecture_f1_score:.4f}, "
                       f"neurons={model.best_architecture_neurons}")
    else:
        if config.get('verbose', True):
            logger.info(f"No best architecture found, keeping current: {model.hidden_neurons} hidden neurons")
    
    return model, overfit, False  # underfit is always False since we assume it until overfit


def _update_training_history(history: Dict[str, List], stage_info: Dict, current_classes: List[int],
                           model: DynamicNetwork, best_history: Dict[str, List], overfit: bool) -> None:
    """
    Update the training history with results from current training stage.
    """
    # Record stage information
    history['stage'].append(stage_info['stage'])
    history['classes_so_far'].append(current_classes.copy())
    history['hidden_neurons'].append(model.hidden_neurons)
    
    # Record performance metrics
    history['val_mcc'].append(best_history['val_mcc'][-1])
    history['val_accuracy'].append(best_history['val_accuracy'][-1])
    history['val_f1'].append(best_history['val_f1'][-1])
    history['training_loss'].append(best_history['train_loss'][-1])
    
    # Record overfitting status
    history['overfit'].append(overfit)


def _prepare_data_for_classes(X_train: np.ndarray, y_train: np.ndarray, 
                              current_classes: List[int], batch_size: int) -> DataLoader:
    """
    Prepare DataLoader for training with specific classes only.
    """
    # Create boolean mask for samples belonging to current classes
    train_mask = np.isin(y_train, current_classes).flatten()
    
    # Filter data to include only current classes
    X_current = X_train[train_mask]
    y_current = y_train[train_mask]

    # Create PyTorch dataset and DataLoader
    dataset = TensorDataset(
        torch.from_numpy(X_current.astype(np.float32)),
        torch.from_numpy(y_current.astype(np.int64))
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _determine_class_order(y_train: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Determine the order of classes for incremental learning.
    """
    # Get unique classes and their sample counts
    classes, counts = np.unique(y_train, return_counts=True)
    
    # Sort classes by sample count (ascending - smallest first)
    class_order = [cls for cls, count in sorted(zip(classes, counts), key=lambda x: x[1])]
    
    # Split into initial (2 smallest) and remaining classes
    initial_classes = class_order[:2]
    remaining_classes = class_order[2:]
    
    return initial_classes, remaining_classes


def _initialise_training_history() -> Dict[str, List]:
    """
    Initialise the training history dictionary for tracking training progress.
    """
    return {
        'stage': [],           # Training stage number
        'classes_so_far': [],  # Classes learned so far
        'hidden_neurons': [],  # Number of hidden neurons at each stage
        'val_accuracy': [],    # Validation accuracy at each stage
        'val_f1': [],          # Validation F1 score at each stage
        'training_loss': [],   # Training loss at each stage
        'val_mcc': [],         # Validation MCC at each stage
        'overfit': [],         # Overfitting status at each stage
        'rollbacks': []        # Architecture rollback information
    }


def run_incremental_learning(X_train: np.ndarray, y_train: np.ndarray, 
                            config: Dict[str, Any], 
                            X_val: Optional[np.ndarray] = None, 
                            y_val: Optional[np.ndarray] = None) -> Tuple[DynamicNetwork, Dict[str, List]]:
    """
    Main incremental learning function that implements the complete training process.
    """
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Determine class order (smallest classes first)
    initial_classes, remaining_classes = _determine_class_order(y_train)
    input_size = X_train.shape[1]
    
    # Initialise model with initial classes
    model = DynamicNetwork(
        input_size=input_size, 
        num_classes=len(initial_classes), 
        mode=config.get('model_mode', 'reinit')
    )
    
    # Initialise training history tracking
    history = _initialise_training_history()
    current_classes = initial_classes.copy()
    
    # Stage 1: Train on initial classes with architecture search
    logger.info(f"Stage 1: Processing initial classes: {initial_classes}")
    
    train_loader = _prepare_data_for_classes(X_train, y_train, current_classes, 
                                           config.get('batch_size', 32))
    
    # Train initial model with architecture search
    model, overfit, underfit = train_model_with_architecture_search(
        model=model,
        train_loader=train_loader,
        val_data=(X_val, y_val),
        current_classes=current_classes,
        config=config,
        stage_info={'stage': 1, 'description': 'Initial classes'},
        history=history
    )
    
    # Stage 2+: Add remaining classes one by one
    for stage_idx, new_class in enumerate(remaining_classes, start=2):
        logger.info(f"Stage {stage_idx}: Adding class {new_class}")
        
        # Add new output class to the model
        current_classes.append(new_class)
        model.add_output_class(k=1)
        
        # Prepare training data for all current classes
        train_loader = _prepare_data_for_classes(X_train, y_train, current_classes,
                                               config.get('batch_size', 32))
        
        # Train with the new class (architecture search may add hidden neurons)
        model, overfit, underfit = train_model_with_architecture_search(
            model=model,
            train_loader=train_loader,
            val_data=(X_val, y_val),
            current_classes=current_classes,
            config=config,
            stage_info={'stage': stage_idx, 'description': f'Added class {new_class}'},
            history=history
        )
    
    return model, history


def _print_fold_results(fold: int, test_results: Dict[str, float]) -> None:
    """
    Print results for a single cross-validation fold.
    """
    print(f"Fold {fold} Results:")
    print(f"  Accuracy: {test_results.get('accuracy', 0):.4f}, F1: {test_results.get('f1_score', 0):.4f}, MCC: {test_results.get('mcc', 0):.4f}")


def run_icl(X_all_train: np.ndarray, y_all_train: np.ndarray,
                   X_all_test: np.ndarray, y_all_test: np.ndarray,
                   config: Dict[str, Any]) -> Dict[str, List]:
    """
    Run incremental learning comparison using cross-validation.
    """
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Initialise stratified k-fold cross-validation
    n_folds = config.get('n_folds', 5)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get('seed', 42))

    # Store results for each fold
    results = {'f1_score': [], 'mcc': [], 'accuracy': [], 'hidden_neurons': []}
    
    fold = 1
    for train_idx, val_idx in skf.split(X_all_train, y_all_train):
        logger.info(f"Running Fold {fold}/{n_folds}")
        
        # Split training data into train/validation for this fold
        X_train, X_val = X_all_train[train_idx], X_all_train[val_idx]
        y_train, y_val = y_all_train[train_idx], y_all_train[val_idx]
        
        # Run incremental learning approach
        model_ft, history_ft = run_incremental_learning(
            X_train, y_train, config, X_val, y_val
        )
        
        # Evaluate on test set using all classes learned
        all_classes = history_ft['classes_so_far'][-1]
        test_results_ft = evaluate_model(model_ft, X_all_test, y_all_test, all_classes)
        
        # Store results for this fold
        results['accuracy'].append(test_results_ft.get('accuracy', 0))
        results['f1_score'].append(test_results_ft.get('f1_score', 0))
        results['mcc'].append(test_results_ft.get('mcc', 0))
        results['hidden_neurons'].append(history_ft['hidden_neurons'][-1])
        
        # Print fold results
        _print_fold_results(fold, test_results_ft)
        fold += 1
    
    return results


def train_model(model: DynamicNetwork, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray, classes: List[int], 
                epochs: int = 50, lr: float = 0.01, batch_size: int = 32, 
                device: str = "cpu", early_stopping: bool = True) -> DynamicNetwork:
    """
    Train a neural network model with mini-batch gradient descent and early stopping.
    """
    
    # Initialise optimiser and scheduler
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.8, patience=10, min_lr=lr*0.01
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialise training tracking variables
    training_history = {'train_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_window = 10
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch_features, batch_labels in train_loader:
            
            optimiser.zero_grad()
            outputs = model(batch_features)
            
            # Map labels to indices
            mapped_labels = torch.tensor([class_to_index[int(label)] for label in batch_labels])
            loss = F.cross_entropy(outputs, mapped_labels)
            
            loss.backward()
            optimiser.step()
            epoch_losses.append(loss.item())
        
        # Calculate average training loss
        avg_epoch_loss = np.mean(epoch_losses)
        training_history['train_loss'].append(avg_epoch_loss)
        
        # Validation phase (if early stopping is enabled)
        if early_stopping:
            # Evaluate on validation set
            val_results = evaluate_model(model, X_val, y_val, classes)
            val_accuracy = val_results.get('accuracy', 0)
            val_f1 = val_results.get('f1_score', 0)
            val_loss = val_results.get('cross_entropy', float('inf'))
            
            training_history['val_accuracy'].append(val_accuracy)
            training_history['val_loss'].append(val_loss)
            training_history['val_f1'].append(val_f1)

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Store best model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            # Check for overfitting after sufficient training
            if epoch >= patience_window:
                _, current_overfit = fit_check(
                    training_history['train_loss'], 
                    training_history['val_loss'], 
                    training_history['val_accuracy'],
                    num_classes=len(classes),
                    patience_window=patience_window
                )
                
                if current_overfit:
                    logger.info(f"Early stopping triggered at epoch {epoch} due to overfitting")
                    break
    
    # Restore best model state if early stopping was used
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def run_base(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Perform k-fold cross-validation with hidden neuron search for baseline comparison.
    """
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Initialise cross-validation
    skf = StratifiedKFold(n_splits=config.get('n_folds', 5), shuffle=True, random_state=config.get('seed', 42))
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    classes = np.unique(y_train)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Starting {config.get('n_folds', 5)}-fold CV with hidden neurons 0-{config.get('max_hidden_neurons', 10)}")
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Cross-validation results storage
    cv_results = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"Processing fold {fold}/{config.get('n_folds', 5)}")

        best_hidden_neurons = 0
        best_score = 0
        best_results = None
        
        # Test different numbers of hidden neurons
        for hidden_units in range(0, config.get('max_hidden_neurons', 10) + 1):
            # Create model with specified number of hidden neurons
            model = DynamicNetwork(input_size=input_size, num_classes=num_classes)
            for _ in range(hidden_units):
                model.add_hidden_neuron()
            
            # Train model on current fold
            model = train_model(
                model=model,
                X_train=X_train[train_idx], 
                y_train=y_train[train_idx],
                X_val=X_train[val_idx], 
                y_val=y_train[val_idx],
                classes=classes,
                epochs=config.get('epochs', 50),
                lr=config.get('learning_rate', 0.01),
                batch_size=config.get('batch_size', 32),
                device=device,
                early_stopping=True
            )

            val_results = evaluate_model(model, X_train[val_idx], y_train[val_idx], classes)
            val_f1 = val_results.get('f1_score', 0)
            if val_f1 > best_score:
                best_score = val_f1
                best_hidden_neurons = hidden_units
                best_results = evaluate_model(model, X_test, y_test, classes)
        
        cv_results.append({'hidden_neurons': best_hidden_neurons, 'mcc': best_results.get('mcc', 0), 'accuracy': best_results.get('accuracy', 0), 'f1_score': best_results.get('f1_score', 0)})
        logger.info(f"Fold {fold} completed")

    # Train final model with optimal architecture on full training data
    logger.info(f"Training final model with {best_hidden_neurons} hidden neurons on full training set")
    
    logger.info(f"Final test results:")
    logger.info(cv_results)
    
    return cv_results


def main_icl(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             config: Dict[str, Any]) -> Dict[str, List]:
    """
    Main function to run the incremental learning experiment.
    """
    logger.info("Starting incremental learning experiment")
    results = run_icl(X_train, y_train, X_test, y_test, config)
    logger.info("Experiment completed successfully")
    return results

def main_base(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              config: Dict[str, Any]) -> Dict[str, List]:
    """
    Run baseline k-fold cross-validation experiment.
    """
    # Run k-fold cross-validation with hidden neuron search
    logger.info("Running baseline k-fold cross-validation with hidden neuron search")
    kfold_results = run_base(
        X_train=X_train, 
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config
    )
    
    # Convert results to the same format as incremental learning
    final_results = {
        'f1_score': [x['f1_score'] for x in kfold_results],
        'mcc': [x['mcc'] for x in kfold_results],
        'accuracy': [x['accuracy'] for x in kfold_results],
        'hidden_neurons': [x['hidden_neurons'] for x in kfold_results]
    }
    return final_results

def run_core(datasets: List[str], seeds: List[int]) -> Tuple[List[float], List[float], List[int], List[int]]:
    """
    Run the complete experiment comparing baseline vs incremental learning across multiple datasets and seeds.
    """
    from evalu import wilcoxon_test
    import time
    
    start_time = time.time()
    config = DEFAULT_CONFIG.to_dict()
    
    # Initialise result storage
    overall_neuron_base, overall_accuracy_base = [], []
    overall_neuron_icl, overall_accuracy_icl = [], []
    ALPHA = 0.05  # Significance level for statistical tests
    
    # Process each dataset
    for dataset in datasets:
        # Initialise per-dataset result storage
        f1_base, mcc_base, accuracy_base, neuron_base = [], [], [], []
        f1_icl, mcc_icl, accuracy_icl, neuron_icl = [], [], [], []

        # Load dataset
        X_train, y_train, X_test, y_test = fetch_data(
            dataset,
            test_size=config.get('test_size', 0.2),
            seed=config.get('seed', 42)
        )
        
        # Test with different random seeds
        for seed in seeds:
            config['seed'] = seed
            logger.setLevel(logging.WARNING)  # Reduce logging verbosity

            # Run baseline approach
            results_base = main_base(X_train, y_train, X_test, y_test, config)
            f1_base.extend(results_base['f1_score'])
            mcc_base.extend(results_base['mcc'])
            accuracy_base.extend([x * 100 for x in results_base['accuracy']])  # Convert to percentage
            neuron_base.extend(results_base['hidden_neurons'])
            
            # Run incremental learning approach
            results_icl = main_icl(X_train, y_train, X_test, y_test, config)
            f1_icl.extend(results_icl['f1_score'])
            mcc_icl.extend(results_icl['mcc'])
            accuracy_icl.extend([x * 100 for x in results_icl['accuracy']])  # Convert to percentage
            neuron_icl.extend(results_icl['hidden_neurons'])

        # Aggregate results across all seeds for this dataset
        overall_neuron_base.extend(neuron_base)
        overall_accuracy_base.extend(accuracy_base)
        overall_neuron_icl.extend(neuron_icl)
        overall_accuracy_icl.extend(accuracy_icl)

        # Perform statistical test
        winner, p_value = wilcoxon_test(f1_icl, f1_base, ALPHA)
        
        # Print results for this dataset
        print("="*50)
        print(f" RESULTS FOR {dataset} ")
        print(f" Winner: {winner}, P-value: {p_value}")
        print(f" F1 scores: {np.mean(f1_base):.3f} +- {np.std(f1_base):.3f}, | | {np.mean(f1_icl):.3f} +- {np.std(f1_icl):.3f}")
        print(f" MCC scores: {np.mean(mcc_base):.3f} +- {np.std(mcc_base):.3f}, | | {np.mean(mcc_icl):.3f} +- {np.std(mcc_icl):.3f}")
        print(f" Accuracy scores: {np.mean(accuracy_base):.3f} +- {np.std(accuracy_base):.3f}, | | {np.mean(accuracy_icl):.3f} +- {np.std(accuracy_icl):.3f}")
        # print(f"${np.mean(accuracy_base):.3f}\\pm{np.std(accuracy_base):.3f}$ & ${np.mean(f1_base):.3f}\\pm{np.std(f1_base):.3f}$ & ${np.mean(mcc_base):.3f}\\pm{np.std(mcc_base):.3f}$ & ${np.mean(accuracy_icl):.3f}\\pm{np.std(accuracy_icl):.3f}$ & ${np.mean(f1_icl):.3f}\\pm{np.std(f1_icl):.3f}$ & ${np.mean(mcc_icl):.3f}\\pm{np.std(mcc_icl):.3f}$")
        print("="*50)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return overall_accuracy_base, overall_accuracy_icl, overall_neuron_base, overall_neuron_icl

if __name__ == "__main__":
    # Define datasets and seeds for the experiment
    DATASETS = ["glass", "sil", "segment", "wine", "yeast"]
    SEEDS = [0, 21, 42, 63, 84]

    # Run the complete experiment
    overall_accuracy_base, overall_accuracy_icl, overall_neuron_base, overall_neuron_icl = run_core(DATASETS, SEEDS)

    # Generate visualisation plots
    from evalu import visualise
    visualise(overall_accuracy_base, overall_accuracy_icl,
              overall_neuron_base, overall_neuron_icl,
              DATASETS, SEEDS, n_folds=DEFAULT_CONFIG.n_folds)