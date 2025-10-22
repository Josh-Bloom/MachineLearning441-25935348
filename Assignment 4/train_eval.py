"""
Evaluate training performance using best hyperparameters from final test evaluation.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Local imports
from data import dataset_loader
from config import SEED

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')


def evaluate_training_performance():
    """
    Evaluate training performance for best parameters from final test evaluation.
    """
    
    best_params_df = pd.read_csv('results/final_test_evaluation.csv')
    print(f'Loaded parameters for {len(best_params_df)} datasets')
    print('\nBest parameters:')
    print(best_params_df[['dataset', 'best_depth', 'best_max_features']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # EVALUATE TRAINING PERFORMANCE FOR EACH DATASET
    # -------------------------------------------------------------------------
    
    # Use same seeds as test evaluation for consistency
    seeds = [0, 1, 2, 3, 4]
    
    rows = []
    for _, row in best_params_df.iterrows():
        dataset = row['dataset']
        best_depth = int(row['best_depth']) if not pd.isna(row['best_depth']) else None
        best_max_features = int(row['best_max_features']) if not pd.isna(row['best_max_features']) else 'sqrt'
        
        print(f'\n{"-"*60}')
        print(f'Evaluating {dataset}...')
        print(f'  Best depth: {best_depth}')
        print(f'  Best max_features: {best_max_features}')
        
        # Storage for metrics across seeds
        accs = []
        f1ms = []
        mccs = []
        
        # Evaluate with each seed for robustness
        for seed in seeds:
            # Load fresh train/test split with this seed
            X_train, X_test, y_train, y_test = dataset_loader(dataset, seed=seed)
            
            # Train model with best hyperparameters (same as test evaluation)
            clf = RandomForestClassifier(
                n_estimators=300,  # Use large ensemble for final model
                max_depth=best_depth,
                max_features=best_max_features,
                random_state=seed,
                n_jobs=-1  # Use all available cores
            )
            clf.fit(X_train, y_train)
            
            # Predict on TRAINING set (key difference from test evaluation)
            y_pred = clf.predict(X_train)
            
            # Compute multiple metrics for comprehensive evaluation
            acc = accuracy_score(y_train, y_pred) * 100  # Percentage
            f1m = f1_score(y_train, y_pred, average='macro')  # Macro F1
            mcc = matthews_corrcoef(y_train, y_pred)  # Matthews correlation
            
            accs.append(acc)
            f1ms.append(f1m)
            mccs.append(mcc)
            
            print(f'    Seed {seed}: {acc:.2f}% accuracy')
        
        # Aggregate metrics across seeds
        rows.append({
            'dataset': dataset,
            'best_depth': best_depth,
            'best_max_features': best_max_features,
            'train_accuracy_percent': np.mean(accs),
            'train_accuracy_std': np.std(accs),
            'train_f1_macro': np.mean(f1ms),
            'train_f1_macro_std': np.std(f1ms),
            'train_mcc': np.mean(mccs),
            'train_mcc_std': np.std(mccs)
        })
    
    training_results_df = pd.DataFrame(rows)
    
    print('\n' + '='*60)
    print('FINAL TRAINING PERFORMANCE SUMMARY')
    print('='*60)
    print(training_results_df.to_string(index=False))


if __name__ == '__main__':
    evaluate_training_performance()
