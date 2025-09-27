import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import copy
from model_util import *
from dynamicNN import *
from data import *

glass_X, glass_y = fetch_data("glass")

def train_model_with_architecture_search(model, train_loader, val_data, current_classes, 
                                       config, stage_info=None, history=None):
    """
    Train model with incremental architecture growth based on underfitting/overfitting detection
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_data: Validation data (X_val, y_val)
        current_classes: List of current classes being learned
        config: Configuration dictionary
        stage_info: Information about current stage
        history: History tracking dictionary
        
    Returns:
        Trained model, overfit flag, underfit flag
    """
    X_val, y_val = val_data
    
    # Initialize training state
    hidden_neurons_added = 0
    max_hidden_neurons = config.get('max_hidden_neurons', 20)
    overfit = False
    underfit = False

    max_epochs = config.get('max_epochs', 100)
    learning_rate = config.get('learning_rate', 0.001)
    patience_window = config.get('patience', 10)  # Reduced for more responsive detection

    class_to_index = {cls: idx for idx, cls in enumerate(current_classes)}
    
    # Main training loop with architecture growth
    # Strategy: Assume underfitting until we detect overfitting, then restore to best architecture
    while hidden_neurons_added < max_hidden_neurons:
        if config.get('verbose', True):
            print(f"Training with {model.hidden_neurons} hidden neurons...")
        
        # Train the model
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Add learning rate scheduler (more conservative)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=patience_window, min_lr=learning_rate*0.01
        )
        
        training_history = {'train_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_history = None
        best_model_state = None
        
        # Reset flags for this architecture
        current_overfit = False
        
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                
                mapped_labels = torch.tensor([class_to_index[int(label)] for label in batch_labels])
                loss = F.cross_entropy(outputs, mapped_labels)
                
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_epoch_loss = np.mean(epoch_losses)
            training_history['train_loss'].append(avg_epoch_loss)
            
            # Evaluate on validation set
            val_results = evaluate_model(model, X_val, y_val, current_classes)
            val_accuracy = val_results.get('accuracy', 0)
            val_f1 = val_results.get('f1_score', 0)
            val_loss = val_results.get('cross_entropy', float('inf'))
            
            training_history['val_accuracy'].append(val_accuracy)
            training_history['val_loss'].append(val_loss)
            training_history['val_f1'].append(val_f1)

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Store best model for this architecture
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_history = copy.deepcopy(training_history)
                # Store the best model state for this architecture
                best_model_state = copy.deepcopy(model.state_dict())

            # Check for overfitting after sufficient training
            if epoch >= patience_window:
                _, current_overfit = fit_check(
                    training_history['train_loss'], 
                    training_history['val_loss'], 
                    training_history['val_accuracy'],
                    num_classes=len(current_classes),
                    patience_window=patience_window
                )
                
                # if config.get('verbose', True) and epoch % 10 == 0:
                #     print(f"  Epoch {epoch}: Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                #     print(f"  Overfit: {current_overfit}")
                
                if current_overfit:
                    if config.get('verbose', True):
                        print(f"  Overfitting detected at epoch {epoch}! Stopping training.")
                    # Note: We don't restore here since we're already tracking the best model
                    # model.restore()  # This works within the same architecture
                    overfit = True
                    break
        
        # Store checkpoint for this architecture
        if best_history is not None and best_model_state is not None:
            # Restore to best model for this architecture before storing checkpoint
            model.load_state_dict(best_model_state)
            model.store_architecture_checkpoint(best_val_loss, best_history['val_accuracy'][-1], model.hidden_neurons)
            
            # if config.get('verbose', True):
            #     print(f"  Stored checkpoint for {model.hidden_neurons} hidden neurons, val_loss: {best_val_loss:.4f}, val_acc: {best_history['val_accuracy'][-1]:.4f}")
        
        # Update history with best results for this architecture
        if history is not None and stage_info is not None and best_history is not None:
            history['stage'].append(stage_info['stage'])
            history['classes_so_far'].append(current_classes.copy())
            history['hidden_neurons'].append(model.hidden_neurons)
            history['val_accuracy'].append(best_history['val_accuracy'][-1])
            history['val_f1'].append(best_history['val_f1'][-1])
            history['training_loss'].append(best_history['train_loss'][-1])
            history['overfit'].append(overfit)
        
        if config.get('verbose', True):
            print(f"  Best validation accuracy: {best_history['val_accuracy'][-1]:.4f}")
            print(f"  Best validation loss: {best_val_loss:.4f}")
        
        # If overfitting detected, stop architecture growth and restore to best architecture
        if overfit:
            # if config.get('verbose', True):
                # print(f"  Overfitting detected! Available checkpoints: {model.list_checkpoints()}")
                # print(f"  Restoring to best architecture...")
            
            # Restore to best architecture found across all attempts
            if model.restore_best_architecture():
                pass
                # if config.get('verbose', True):
                #     print(f"  Restored to best architecture with {model.hidden_neurons} hidden neurons")
            else:
                pass
                # if config.get('verbose', True):
                #     print(f"  No previous best architecture found, keeping current: {model.hidden_neurons} hidden neurons")
            break
        
        # Assume underfitting - add hidden neuron and continue
        # if config.get('verbose', True):
        #     print(f"  Assuming underfitting. Adding hidden neuron... (was {model.hidden_neurons})")
        
        model.add_hidden_neuron()
        hidden_neurons_added += 1
        
        # if config.get('verbose', True):
        #     print(f"  Now has {model.hidden_neurons} hidden neurons")
        
        # Adjust training parameters for fine-tuning
        if config.get('model_mode', 'reinit') == 'preserve':
            max_epochs = config.get('fine_max_epochs', 30)
            learning_rate = config.get('fine_learning_rate', 0.0005)
            patience_window = config.get('fine_patience', 5)
    
    # Always restore to best architecture at the end (regardless of how we exited the loop)
    # if config.get('verbose', True):
    #     print(f"  Final step: Restoring to best architecture...")
    
    if model.restore_best_architecture():
        if config.get('verbose', True):
            # print(f"  Restored to best architecture with {model.hidden_neurons} hidden neurons")
            print(f"  Best architecture metrics: loss={model.best_architecture_loss:.4f}, accuracy={model.best_architecture_accuracy:.4f}")
    else:
        if config.get('verbose', True):
            print(f"  No best architecture found, keeping current: {model.hidden_neurons} hidden neurons")
    
    return model, overfit, False  # underfit is always False now since we assume it until overfit


def run_incremental_learning(X_train, y_train, config, X_val=None, y_val=None):
    """Main incremental learning function with unified training approach"""
    set_seed(config.get('seed', 42))
    
    classes, counts = np.unique(y_train, return_counts=True)
    class_order = [cls for cls, count in sorted(zip(classes, counts), key=lambda x: x[1])]
    
    initial_classes = class_order[:2]
    remaining_classes = class_order[2:]
    
    input_size = X_train.shape[1]
    model = DynamicNetwork(input_size=input_size, num_classes=len(initial_classes), mode=config.get('model_mode', 'reinit'))
    # ewc = EWCRegularizer(model) if approach == 'ewc' else None
    
    history = {
        'stage': [], 'classes_so_far': [], 'hidden_neurons': [],
        'val_accuracy': [], 'val_f1': [], 'training_loss': [],
        'overfit': [], 'rollbacks': []
    }
    
    current_classes = initial_classes.copy()
    # checkpoint_manager = ModelCheckpoint()
    
    # Process initial classes
    if config.get('verbose', True):
        print(f"\n=== Stage 1: Processing initial classes: {initial_classes} ===")
    
    train_mask = np.isin(y_train, current_classes).flatten()
    X_current = X_train[train_mask]
    y_current = y_train[train_mask]

    dataset = TensorDataset(
        torch.from_numpy(X_current.astype(np.float32)),
        torch.from_numpy(y_current.astype(np.int64))
    )
    train_loader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    
    # Save initial state
    # checkpoint_manager.save_checkpoint(model, model.hidden_neurons, current_classes)
    
    # Train initial model with architecture search
    model, overfit, underfit = train_model_with_architecture_search(
        model=model,
        train_loader=train_loader,
        val_data=(X_val, y_val),
        current_classes=current_classes,
        config=config,
        # ewc=ewc,
        # checkpoint_manager=checkpoint_manager,
        stage_info={'stage': 1, 'description': 'Initial classes'},
        history=history
    )
    
    # Process remaining classes
    for stage_idx, new_class in enumerate(remaining_classes, start=2):
        if config.get('verbose', True):
            print(f"\n=== Stage {stage_idx}: Adding class {new_class} ===")
        
        # Save checkpoint before adding new class
        # checkpoint_manager.save_checkpoint(model, model.hidden_neurons, current_classes)
        
        # Add new class
        current_classes.append(new_class)
        model.add_output_class(k=1)
        
        # Prepare training data for current classes
        train_mask = np.isin(y_train, current_classes).flatten()
        X_current = X_train[train_mask]
        y_current = y_train[train_mask]
        
        dataset = TensorDataset(
            torch.from_numpy(X_current.astype(np.float32)),
            torch.from_numpy(y_current.astype(np.int64))
        )
        train_loader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)
        
        # Train with the new class
        model, overfit, underfit = train_model_with_architecture_search(
            model=model,
            train_loader=train_loader,
            val_data=(X_val, y_val),
            current_classes=current_classes,
            config=config,
            # ewc=ewc,
            # checkpoint_manager=checkpoint_manager,
            stage_info={'stage': stage_idx, 'description': f'Added class {new_class}'},
            history=history
        )
        
        # Handle overfitting after class addition
        # if overfit:
        #     if config.get('verbose', True):
        #         print("  Overfitting detected after class addition! Rolling back...")
            
        #     # Restore previous state
        #     prev_hidden, prev_classes, _ = checkpoint_manager.restore_previous(model)
        #     if prev_hidden is not None:
        #         current_classes = prev_classes
        #         if history is not None:
        #             history['rollbacks'].append(('class_addition', new_class))
        #         if config.get('verbose', True):
        #             print(f"  Rolled back class {new_class}, current classes: {current_classes}")
        #     else:
        #         if config.get('verbose', True):
        #             print("  Cannot rollback further")
    
    return model, history


import pandas as pd
def run_comparison(X, y, config, n_folds=3):
    """
    Compare fine-tuning and EWC approaches using cross-validation.
    """
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get('seed', 42))

    X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=config.get('seed', 42)
    )

    # Store results for each approach
    results = {
        'fine_tune': {'val_accuracy': [], 'val_f1': [], 'hidden_neurons': []}
    }
    
    fold = 1
    for train_idx, val_idx in skf.split(X_all_train, y_all_train):
        print(f"\n{'='*50}")
        print(f"Running Fold {fold}/{n_folds}")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X_all_train[train_idx], X_all_train[val_idx]
        y_train, y_val = y_all_train[train_idx], y_all_train[val_idx]
        
        # Run fine-tuning approach
        # print("Running fine-tuning approach...")
        model_ft, history_ft = run_incremental_learning(
            X_train, y_train, config, X_val, y_val
        )
        
        # Run EWC approach
        # print("Running EWC approach...")
        # model_ewc, history_ewc = run_incremental_learning(
        #     X_train, y_train, config, 'ewc', X_val, y_val
        # )
        
        # Evaluate on test set
        test_results_ft = evaluate_model(model_ft, X_all_test, y_all_test, history_ft['classes_so_far'][-1])
        # test_results_ewc = evaluate_model(model_ewc, X_test, y_test, history_ewc['classes_so_far'][-1])
        
        # Debug: Show the discrepancy
        # print(f"\nDEBUG - Fold {fold} Evaluation:")
        # print(f"  Model final hidden neurons: {model_ft.hidden_neurons}")
        # print(f"  History reported accuracy: {history_ft['val_accuracy'][-1]:.4f}")
        # print(f"  Actual validation accuracy: {val_results_ft.get('accuracy', 0):.4f}")
        # print(f"  Difference: {abs(history_ft['val_accuracy'][-1] - val_results_ft.get('accuracy', 0)):.4f}")
        
        # Store results
        results['fine_tune']['val_accuracy'].append(test_results_ft.get('accuracy', 0))  # Use actual evaluation
        results['fine_tune']['val_f1'].append(test_results_ft.get('f1_score', 0))  # Use actual evaluation
        results['fine_tune']['hidden_neurons'].append(history_ft['hidden_neurons'][-1])
        
        # results['ewc']['val_accuracy'].append(history_ewc['val_accuracy'][-1])
        # results['ewc']['val_f1'].append(history_ewc['val_f1'][-1])
        # results['ewc']['hidden_neurons'].append(history_ewc['hidden_neurons'][-1])
        
        print(f"Fold {fold} Results:")
        print(f"  Fine-tune - Accuracy: {test_results_ft.get('accuracy', 0):.4f}, F1: {test_results_ft.get('f1_score', 0):.4f}")
        # print(f"  EWC       - Accuracy: {test_results_ewc.get('accuracy', 0):.4f}, F1: {test_results_ewc.get('f1_score', 0):.4f}")
        
        fold += 1
    
    # Calculate average performance
    for approach in results:
        results[approach]['mean_accuracy'] = np.mean(results[approach]['val_accuracy'])
        results[approach]['mean_f1'] = np.mean(results[approach]['val_f1'])
        results[approach]['mean_hidden_neurons'] = np.mean(results[approach]['hidden_neurons'])
        results[approach]['std_accuracy'] = np.std(results[approach]['val_accuracy'])
        results[approach]['std_f1'] = np.std(results[approach]['val_f1'])
    
    print(f"\n{'='*50}")
    print("Comparison Summary:")
    print(f"{'='*50}")
    print(f"Fine-tune - Accuracy: {results['fine_tune']['mean_accuracy']:.4f} (±{results['fine_tune']['std_accuracy']:.4f}), "
          f"F1: {results['fine_tune']['mean_f1']:.4f} (±{results['fine_tune']['std_f1']:.4f}), "
          f"Hidden Neurons: {results['fine_tune']['mean_hidden_neurons']:.1f}")
    # print(f"EWC       - Accuracy: {results['ewc']['mean_accuracy']:.4f} (±{results['ewc']['std_accuracy']:.4f}), "
    #       f"F1: {results['ewc']['mean_f1']:.4f} (±{results['ewc']['std_f1']:.4f}), "
    #       f"Hidden Neurons: {results['ewc']['mean_hidden_neurons']:.1f}")

    return results

# Example configuration
default_config = {
    'seed': 42,
    'batch_size': 16,
    'learning_rate': 0.002,
    'fine_learning_rate': 0.0005,
    'weight_decay': 1e-5,
    'max_epochs': 100,  # Back to original
    'fine_max_epochs': 30,
    'patience': 10,  # Back to original
    'fine_patience': 5,
    'max_hidden_neurons': 64,  # Back to original
    'verbose': True,
    'model_mode': 'preserve'
}

results = run_comparison(glass_X, glass_y, default_config, n_folds=5)
# print(results)


# ====== Training with mini-batches ======
def train_model(model, X_train, y_train, X_val, y_val, classes, epochs=50, lr=0.01, batch_size=32, device="cpu", early_stopping=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_accs = []
    val_accs = []

    training_history = {'train_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_history = None
    best_model_state = None
    patience_window = 10
    # Add learning rate scheduler (more conservative)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=patience_window, min_lr=lr*0.01
    )

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            class_to_index = {cls: idx for idx, cls in enumerate(classes)}
            mapped_labels = torch.tensor([class_to_index[int(label)] for label in batch_labels])
            
            loss = F.cross_entropy(outputs, mapped_labels)
            
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if early_stopping:
            avg_epoch_loss = np.mean(epoch_losses)
            training_history['train_loss'].append(avg_epoch_loss)
            
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

            # Store best model for this architecture
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_history = copy.deepcopy(training_history)
                # Store the best model state for this architecture
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
                    break
        val_acc = evaluate_model(model, X_val, y_val, classes).get('accuracy', 0)
        return val_acc


# ====== Cross-validation + hidden neuron search ======
def evaluate_with_kfold(X, y, max_hidden=10, n_splits=5, epochs=50, batch_size=16):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    input_size = X.shape[1]
    num_classes = len(np.unique(y))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_acc = []
        for hidden_units in range(0, max_hidden + 1):
            model = DynamicNetwork(input_size=input_size, num_classes=num_classes).to(device)
            for _ in range(hidden_units):
                model.add_hidden_neuron()
            acc = train_model(
                model,
                X_train[train_idx], y_train[train_idx],
                X_train[val_idx], y_train[val_idx],
                np.unique(y),
                epochs=epochs,
                lr=0.01,
                batch_size=batch_size,
                device=device
            )
            fold_acc.append(acc)
        results.append(fold_acc)
        print("--------------------------------")
    print(f"CV Accuracy: {np.round(np.mean(results, axis=0), 4)}")
    
    best_hidden = np.argmax(np.mean(results, axis=0))
    print(f"\nBest hidden units: {best_hidden} with accuracy {np.round(np.mean(results, axis=0)[best_hidden], 4)}")

    model = DynamicNetwork(input_size=input_size, num_classes=num_classes).to(device)
    for _ in range(best_hidden):
        model.add_hidden_neuron()
    acc = train_model(
            model,
            X_train, y_train,
            X_test, y_test,
            np.unique(y),
            epochs=epochs,
            lr=0.01,
            batch_size=batch_size,
            device=device
        )
    print(acc)
    return results

# _ = evaluate_with_kfold(glass_X, glass_y, max_hidden=64, n_splits=5, epochs=100)