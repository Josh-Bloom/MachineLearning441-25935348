"""
Core script for Random Forest experiments.

Coordinates dataset loading, experimental phases, plotting, and final evaluation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Local imports
from utils import set_seed
from data import dataset_loader
from experiments import *
from config import SEED, DEFAULT_MAX_FEATURES, SEEDS, DEFAULT_DEPTH_VALUES_SLIM

def run_experiments(datasets: List[str], phases: List[str], seeds: List[int] = None) -> None:
    """
    Run experimental phases across datasets.
    """
    # Use configured seeds if none provided
    if seeds is None:
        seeds = SEEDS
    
    # Set global random seed for reproducibility
    set_seed(SEED)

    # Collect DataFrames from each phase for combined plotting
    phase_a_dfs, phase_b_dfs, phase_c_dfs, phase_d_dfs = [], [], [], []

    # Track best hyperparameters discovered for each dataset
    best_params_by_dataset: Dict[str, Dict[str, any]] = {}
    # -------------------------------------------------------------------------
    # RUN EXPERIMENTS FOR EACH DATASET
    # -------------------------------------------------------------------------
    for ds in datasets:
        print('\n' + '='*60)
        print(f'Running experiments for dataset: {ds}')
        
        # Load dataset with train/test split
        X_train, X_test, y_train, y_test = dataset_loader(ds)
        print(f'Loaded dataset {ds}: X_train.shape={X_train.shape}, '
              f'y_train.shape={y_train.shape}, classes={np.unique(y_train)}')

        best_depth = None  # Will be set by Phase A if run
        
        # ---------------------------------------------------------------------
        # PHASE A: DEPTH SWEEP
        # ---------------------------------------------------------------------
        if 'A' in phases:
            print(f'\n[{ds}] Starting Phase A: Depth Sweep')
            df_a, best_depth = phase_a_depth_sweep(ds, X_train, y_train, seeds=seeds)
            phase_a_dfs.append(df_a)
            print(f'[{ds}] Phase A complete. Best depth: {best_depth}')

        # ---------------------------------------------------------------------
        # PHASE B: MAX_FEATURES SWEEP AT MULTIPLE DEPTHS
        # ---------------------------------------------------------------------
        if 'B' in phases:
            print(f'\n[{ds}] Starting Phase B: Max Features Sweep (multiple depths)')
            
            # Start with representative depth values
            depth_values = DEFAULT_DEPTH_VALUES_SLIM.copy()
            
            # Include best depth from Phase A if not already present
            if best_depth is not None and best_depth not in depth_values:
                depth_values = [best_depth] + depth_values
            
            # Run Phase B experiments
            df_b = phase_b_max_features_sweep(ds, X_train, y_train,
                                              depth_values=depth_values, seeds=seeds)
            phase_b_dfs.append(df_b)
            
            # Select best max_features at the best depth
            if best_depth is not None:
                # Filter to just the best depth
                sub = df_b[df_b['max_depth'] == best_depth]
                if not sub.empty:
                    # Average test accuracy over seeds for each max_features
                    sub_avg = sub.groupby('max_features')['test_accuracy_mean'].mean()
                    best_mf = sub_avg.idxmax()
                else:
                    # Fallback: average over all depths if best_depth has no results
                    avg_by_mf = df_b.groupby('max_features')['test_accuracy_mean'].mean()
                    best_mf = avg_by_mf.idxmax()
                
                # Store for final evaluation
                best_params_by_dataset.setdefault(ds, {})['best_max_features'] = best_mf
                print(f'[{ds}] Phase B complete. Best max_features: {best_mf}')

        # ---------------------------------------------------------------------
        # PHASE D: MAX_FEATURES SWEEP AT SINGLE BEST DEPTH
        # ---------------------------------------------------------------------
        if 'D' in phases:
            print(f'\n[{ds}] Starting Phase D: Max Features Sweep (best depth only)')
            
            # Ensure we have a best depth (run Phase A if needed)
            if best_depth is None:
                print(f'[{ds}] Phase D requires best depth. Running Phase A first...')
                df_a2, best_depth = phase_a_depth_sweep(ds, X_train, y_train, seeds=seeds)
                phase_a_dfs.append(df_a2)
                print(f'[{ds}] Phase A complete. Best depth: {best_depth}')
            
            # Run Phase D experiments
            df_d = phase_d_max_features_at_best_depth(ds, X_train, y_train, 
                                                     best_depth=best_depth, seeds=seeds)
            phase_d_dfs.append(df_d)
            
            # Select best max_features (averaged over seeds)
            avg_by_mf = df_d.groupby('max_features')['test_accuracy_mean'].mean()
            best_mf_d = avg_by_mf.idxmax()
            
            # Store for final evaluation (overrides Phase B if both run)
            best_params_by_dataset.setdefault(ds, {})['best_max_features'] = best_mf_d
            print(f'[{ds}] Phase D complete. Best max_features: {best_mf_d}')

        # ---------------------------------------------------------------------
        # PHASE C: DEPTH VS ENSEMBLE SIZE INTERACTION
        # ---------------------------------------------------------------------
        if 'C' in phases:
            print(f'\n[{ds}] Starting Phase C: Depth vs Ensemble Size')
            
            # Start with slim depth values for efficiency
            depth_values_c = DEFAULT_DEPTH_VALUES_SLIM.copy()
            
            # Include best depth from Phase A if available
            if best_depth is not None and best_depth not in depth_values_c:
                depth_values_c = [best_depth] + depth_values_c
            
            # Run Phase C experiments
            df_c = phase_c_depth_vs_n_estimators(ds, X_train, y_train,
                                                 depth_values=depth_values_c,
                                                 seeds=seeds)
            phase_c_dfs.append(df_c)
            print(f'[{ds}] Phase C complete.')

    # -------------------------------------------------------------------------
    # GENERATE COMBINED PLOTS ACROSS ALL DATASETS
    # -------------------------------------------------------------------------

    print('\n' + '='*60)
    print('GENERATING COMBINED PLOTS')
    print('='*60)
    
    if phase_a_dfs:
        print('Creating Phase A combined plot (depth sweep)...')
        plot_phase_a_combined(phase_a_dfs)
    
    if phase_b_dfs:
        print('Creating Phase B combined plot (max_features at multiple depths)...')
        plot_phase_b_combined(phase_b_dfs)
    
    if phase_c_dfs:
        print('Creating Phase C combined plot (depth vs ensemble size)...')
        plot_phase_c_combined(phase_c_dfs)
    
    if phase_d_dfs:
        print('Creating Phase D combined plot (max_features at best depth)...')
        plot_phase_d_combined(phase_d_dfs)

    # -------------------------------------------------------------------------
    # FINAL EVALUATION ON TEST SETS
    # -------------------------------------------------------------------------
    print('\n' + '='*60)
    print('FINAL TEST SET EVALUATION')
    print('='*60)
    print('Evaluating best hyperparameters on test sets...')
    
    rows = []
    for ds in datasets:
        print(f'\nEvaluating {ds}...')
        
        # Storage for metrics across seeds
        accs = []
        f1ms = []
        mccs = []
        
        # Evaluate with each seed for robustness
        for seed in seeds:
            # Load fresh train/test split with this seed
            X_train, X_test, y_train, y_test = dataset_loader(ds, seed=seed)
            
            # Retrieve best hyperparameters for this dataset
            params = best_params_by_dataset.get(ds, {})
            
            # Recover best depth from Phase A results
            best_depth = None
            if phase_a_dfs:
                # Filter Phase A results for this dataset
                df_a_ds = pd.concat([df[df['dataset'] == ds] for df in phase_a_dfs 
                                   if not df[df['dataset'] == ds].empty], ignore_index=True)
                if not df_a_ds.empty:
                    # Find depth with highest test accuracy
                    best_idx = df_a_ds['test_accuracy_mean'].astype(float).values.argmax()
                    best_depth = df_a_ds.iloc[best_idx]['max_depth']
            
            # Get best max_features (from Phase B or D, or use default)
            best_mf = params.get('best_max_features', DEFAULT_MAX_FEATURES)
            
            # Train final model with best hyperparameters
            clf = RandomForestClassifier(
                n_estimators=300,  # Use large ensemble for final model
                max_depth=None if pd.isna(best_depth) else (int(best_depth) if best_depth is not None else None),
                max_features=best_mf,
                random_state=seed,
                n_jobs=-1  # Use all available cores
            )
            clf.fit(X_train, y_train)
            
            # Predict on held-out test set
            y_pred = clf.predict(X_test)
            
            # Compute multiple metrics for comprehensive evaluation
            acc = accuracy_score(y_test, y_pred) * 100  # Percentage
            f1m = f1_score(y_test, y_pred, average='macro')  # Macro F1
            mcc = matthews_corrcoef(y_test, y_pred)  # Matthews correlation
            
            accs.append(acc)
            f1ms.append(f1m)
            mccs.append(mcc)
        
        # Aggregate metrics across seeds
        rows.append({
            'dataset': ds,
            'best_depth': best_depth,
            'best_max_features': best_mf,
            'test_accuracy_percent': np.mean(accs),
            'test_accuracy_std': np.std(accs),
            'test_f1_macro': np.mean(f1ms),
            'test_f1_macro_std': np.std(f1ms),
            'test_mcc': np.mean(mccs),
            'test_mcc_std': np.std(mccs)
        })
        
        print(f'  {ds}: {np.mean(accs):.2f}% +- {np.std(accs):.2f}% accuracy')
    
    # Save final results to CSV
    if rows:
        final_df = pd.DataFrame(rows)
        final_df.to_csv('results/final_test_evaluation.csv', index=False)
        print('\nResults saved to: results/final_test_evaluation.csv')

    print('\n' + '='*60)
    print('ALL EXPERIMENTS COMPLETED SUCCESSFULLY')
    print('='*60)


if __name__ == '__main__':
    # Suppress sklearn warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run all phases on all datasets with multiple seeds
    run_experiments(
        datasets=['iris', 'wine', 'letter'],
        phases=['A', 'B', 'C', 'D'],
        seeds=[0, 1, 2, 3, 4]
    )