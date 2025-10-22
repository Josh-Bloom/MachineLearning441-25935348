"""
Experiment phases for Random Forest hyperparameter exploration.

Phase A: Sweep tree depth with fixed n_estimators and max_features
Phase B: Sweep max_features at multiple depths (including best from A)
Phase C: Sweep both depth and n_estimators to study interaction
Phase D: Sweep max_features at single best depth from A
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional, Any, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from utils import evaluate_config, find_max_depth, find_max_features
from config import *

def _dataset_markers() -> Dict[str, str]:
    """
    Return marker styles for each dataset.
    """
    return {
        'iris': 'o',         # Circle
        'wine': 's',         # Square
        'letter': 'D'        # Diamond
    }


def _dataset_linestyles() -> Dict[str, str]:
    """
    Return line styles for each dataset.
    """
    return {
        'iris': '-',         # Solid
        'wine': '--',        # Dashed
        'letter': ':'        # Dotted
    }

# ============================================================================
# PHASE A: DEPTH SWEEP
# ============================================================================

def phase_a_depth_sweep(
    dataset_name: str, 
    X: np.ndarray, 
    y: np.ndarray,
    depth_values: List[Optional[int]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seeds: List[int] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Sweep over max_depth values to find optimal tree complexity.
    """
    # Set default hyperparameter ranges
    if depth_values is None:
        depth_values = DEFAULT_DEPTH_VALUES
    
    if seeds is None:
        seeds = [SEED]

    # Container for all experimental results
    records = []
    total = len(depth_values) * len(seeds)
    counter = 0
    
    # Outer loop: iterate over random seeds for robustness
    for seed in seeds:
        from sklearn.model_selection import StratifiedKFold
        
        # Create fresh CV splitter for this seed
        cv_obj = StratifiedKFold(n_splits=CV_FOLDS, random_state=seed, shuffle=True)
            
        # Inner loop: iterate over depth values
        for depth in depth_values:
            counter += 1
            print(f'[{dataset_name}] Phase A: {counter}/{total} - seed={seed}, max_depth={depth}')
            
            # Create Random Forest with specified hyperparameters
            clf = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=depth,
                max_features=DEFAULT_MAX_FEATURES, 
                random_state=seed, 
                n_jobs=N_JOBS
            )
            
            # Evaluate using cross-validation
            stats, rf_estimators = evaluate_config(X, y, clf, cv=cv_obj)
            
            # Build record for this configuration
            rec = {
                'dataset': dataset_name,
                'phase': 'A_depth_sweep',
                'seed': seed,
                # For depth=None, extract actual depth from fitted trees
                'max_depth': find_max_depth(rf_estimators) if depth is None else depth,
                'max_features': find_max_features(DEFAULT_MAX_FEATURES, X.shape[1]),
                'n_estimators': n_estimators,
            }
            rec.update(stats)  # Add all CV metrics
            records.append(rec)

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # BEST DEPTH SELECTION
    # -------------------------------------------------------------------------
    # Strategy: Balance test accuracy (primary) with generalisation (secondary)
    
    df_eval = df.copy()
    df_eval['depth_numeric'] = df_eval['max_depth']
    
    # Aggregate across seeds: for each depth, compute mean train/test accuracy
    depth_stats = df_eval.groupby('max_depth').agg({
        'test_accuracy_mean': 'mean',
        'train_accuracy_mean': 'mean'
    })
    
    # Generalisation gap: large gap suggests overfitting
    depth_stats['gap'] = depth_stats['train_accuracy_mean'] - depth_stats['test_accuracy_mean']
    
    # Normalise metrics to [0, 1] for fair weighted combination
    test_acc_normalised = (depth_stats['test_accuracy_mean'] - depth_stats['test_accuracy_mean'].min()) / \
                         (depth_stats['test_accuracy_mean'].max() - depth_stats['test_accuracy_mean'].min())
    gap_normalised = (depth_stats['gap'] - depth_stats['gap'].min()) / \
                    (depth_stats['gap'].max() - depth_stats['gap'].min())
    
    # Combined score: maximise test accuracy (70%), minimise gap (30%)
    # This favors high test accuracy while avoiding severe overfitting
    depth_stats['score'] = test_acc_normalised - 0.3 * gap_normalised
    
    best_depth = depth_stats['score'].idxmax()
    print(f'[{dataset_name}] Best depth selected: {best_depth} '
          f'(test_acc: {depth_stats.loc[best_depth, "test_accuracy_mean"]:.2f}%, '
          f'gap: {depth_stats.loc[best_depth, "gap"]:.2f}%)')
    
    return df, best_depth


# ============================================================================
# PHASE B: MAX FEATURES SWEEP (AT MULTIPLE DEPTHS)
# ============================================================================

def phase_b_max_features_sweep(
    dataset_name: str, 
    X: np.ndarray, 
    y: np.ndarray,
    depth_values: List[Optional[int]],
    max_features_values: List[Any] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seeds: List[int] = None
) -> pd.DataFrame:
    """
    Sweep over max_features at multiple tree depths.
    """
    if max_features_values is None:
        max_features_values = DEFAULT_MAX_FEATURES_VALUES
    
    if seeds is None:
        seeds = [SEED]

    records = []
    total = len(depth_values) * len(max_features_values) * len(seeds)
    counter = 0
    
    # Triple nested loop: seeds × depths × max_features
    for seed in seeds:
        from sklearn.model_selection import StratifiedKFold
        cv_obj = StratifiedKFold(n_splits=CV_FOLDS, random_state=seed, shuffle=True)
            
        for depth in depth_values:
            for mf in max_features_values:
                counter += 1
                print(f'[{dataset_name}] Phase B: {counter}/{total} - '
                      f'seed={seed}, max_depth={depth}, max_features={mf}')
                
                # Create and evaluate Random Forest
                clf = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=depth,
                    max_features=mf, 
                    random_state=seed, 
                    n_jobs=N_JOBS
                )
                stats, rf_estimators = evaluate_config(X, y, clf, cv=cv_obj)
                
                # Record results
                rec = {
                    'dataset': dataset_name,
                    'phase': 'B_max_features_sweep',
                    'seed': seed,
                    'max_depth': find_max_depth(rf_estimators) if depth is None else depth,
                    'max_features': find_max_features(mf, X.shape[1]),
                    'n_estimators': n_estimators,
                }
                rec.update(stats)
                records.append(rec)

    return pd.DataFrame(records)


# ============================================================================
# PHASE C: DEPTH VS ENSEMBLE SIZE INTERACTION
# ============================================================================

def phase_c_depth_vs_n_estimators(
    dataset_name: str, 
    X: np.ndarray, 
    y: np.ndarray,
    depth_values: List[Optional[int]] = None,
    n_estimators_list: List[int] = None,
    seeds: List[int] = None
) -> pd.DataFrame:
    """
    Study interaction between tree depth and ensemble size.
    """
    if depth_values is None:
        depth_values = DEFAULT_DEPTH_VALUES_SLIM
    if n_estimators_list is None:
        n_estimators_list = DEFAULT_N_ESTIMATORS_VALUES
    
    if seeds is None:
        seeds = [SEED]

    records = []
    total = len(depth_values) * len(n_estimators_list) * len(seeds)
    counter = 0
    
    # Triple nested loop: seeds × depths × ensemble sizes
    for seed in seeds:
        from sklearn.model_selection import StratifiedKFold
        cv_obj = StratifiedKFold(n_splits=CV_FOLDS, random_state=seed, shuffle=True)
            
        for depth in depth_values:
            for n_est in n_estimators_list:
                counter += 1
                print(f'[{dataset_name}] Phase C: {counter}/{total} - '
                      f'seed={seed}, depth={depth}, n_estimators={n_est}')
                
                # Create and evaluate Random Forest
                clf = RandomForestClassifier(
                    n_estimators=n_est, 
                    max_depth=depth,
                    max_features=DEFAULT_MAX_FEATURES, 
                    random_state=seed, 
                    n_jobs=N_JOBS
                )
                stats, rf_estimators = evaluate_config(X, y, clf, cv=cv_obj)
                
                # Record results
                rec = {
                    'dataset': dataset_name,
                    'phase': 'C_depth_vs_n_estimators',
                    'seed': seed,
                    'max_depth': find_max_depth(rf_estimators) if depth is None else depth,
                    'max_features': find_max_features(DEFAULT_MAX_FEATURES, X.shape[1]),
                    'n_estimators': n_est,
                }
                rec.update(stats)
                records.append(rec)

    return pd.DataFrame(records)


# ============================================================================
# PHASE D: MAX FEATURES SWEEP (AT SINGLE BEST DEPTH)
# ============================================================================

def phase_d_max_features_at_best_depth(
    dataset_name: str, 
    X: np.ndarray, 
    y: np.ndarray,
    best_depth: Optional[int],
    max_features_values: List[Any] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seeds: List[int] = None
) -> pd.DataFrame:
    """
    Sweep over max_features at single best depth.
    """

    if max_features_values is None:
        max_features_values = DEFAULT_MAX_FEATURES_VALUES
    
    if seeds is None:
        seeds = [SEED]

    records = []
    total = len(max_features_values) * len(seeds)
    counter = 0
    
    # Nested loop: seeds × max_features (depth is fixed)
    for seed in seeds:
        from sklearn.model_selection import StratifiedKFold
        cv_obj = StratifiedKFold(n_splits=CV_FOLDS, random_state=seed, shuffle=True)
            
        for mf in max_features_values:
            counter += 1
            print(f'[{dataset_name}] Phase D: {counter}/{total} - '
                  f'seed={seed}, max_depth={best_depth}, max_features={mf}')
            
            # Create and evaluate Random Forest at best depth
            clf = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=best_depth,
                max_features=mf, 
                random_state=seed, 
                n_jobs=N_JOBS
            )
            stats, rf_estimators = evaluate_config(X, y, clf, cv=cv_obj)
            
            # Record results
            rec = {
                'dataset': dataset_name,
                'phase': 'D_best_depth_max_features_sweep',
                'seed': seed,
                'max_depth': find_max_depth(rf_estimators) if best_depth is None else best_depth,
                'max_features': find_max_features(mf, X.shape[1]),
                'n_estimators': n_estimators,
            }
            rec.update(stats)
            records.append(rec)

    return pd.DataFrame(records)


# ============================================================================
# COMBINED PLOTTING ACROSS DATASETS
# ============================================================================

def plot_phase_a_combined(dfs: List[pd.DataFrame]) -> None:
    """
    Plot depth vs accuracy across all datasets.
    """
    try:
        # Combine all datasets into single DataFrame
        combined = pd.concat(dfs, ignore_index=True)
        
        # Two-level aggregation:
        # Results are already averaged over CV folds (from evaluate_config)
        # Now average over multiple random seeds for robust estimates
        agg_funcs = {
            'test_accuracy_mean': ['mean', 'std'],
            'train_accuracy_mean': ['mean', 'std']
        }
        aggregated = combined.groupby(['dataset', 'max_depth']).agg(agg_funcs).reset_index()
        
        # Flatten MultiIndex columns for easier access
        aggregated.columns = ['dataset', 'max_depth', 'test_accuracy_mean', 'test_accuracy_std', 
                              'train_accuracy_mean', 'train_accuracy_std']
        
        # Set up styling
        markers = _dataset_markers()
        linestyles = _dataset_linestyles()
        # Distinguish train/test by color; dataset by marker+linestyle
        palette = sns.color_palette('colorblind')
        metric_colors = {'test': palette[0], 'train': palette[1]}  # Blue and orange
        
        plt.figure(figsize=(9, 6))
        
        # Plot each dataset separately
        for ds, sub in aggregated.groupby('dataset'):
            sub = sub.sort_values('max_depth')
            
            # Plot test (validation) accuracy
            plt.errorbar(sub['max_depth'], sub['test_accuracy_mean'], 
                        yerr=sub['test_accuracy_std'], 
                        fmt='none', ecolor=metric_colors['test'], alpha=0.5, 
                        capsize=3, label='_nolegend_')
            plt.plot(sub['max_depth'], sub['test_accuracy_mean'], 
                    marker=markers.get(ds, 'o'), linestyle=linestyles.get(ds, '-'), 
                    color=metric_colors['test'], alpha=0.8, label='_nolegend_')
            
            # Plot training accuracy
            plt.errorbar(sub['max_depth'], sub['train_accuracy_mean'], 
                        yerr=sub['train_accuracy_std'], 
                        fmt='none', ecolor=metric_colors['train'], alpha=0.5, 
                        capsize=3, label='_nolegend_')
            plt.plot(sub['max_depth'], sub['train_accuracy_mean'], 
                    marker=markers.get(ds, 'o'), linestyle=linestyles.get(ds, '-'), 
                    color=metric_colors['train'], alpha=0.8, label='_nolegend_')
        
        plt.xlabel('Maximum Tree Depth')
        plt.ylabel('Average Accuracy (%)')
        plt.grid(True, alpha=0.4)
        
        # Build custom legend showing all dataset-metric combinations
        from matplotlib.lines import Line2D
        handles = []
        for ds in combined['dataset'].unique():
            handles.append(Line2D([0], [0], color=metric_colors['test'], 
                                 linestyle=linestyles.get(ds, '-'), 
                                 marker=markers.get(ds, 'o'), 
                                 label=f'{ds} (test)'))
            handles.append(Line2D([0], [0], color=metric_colors['train'], 
                                 linestyle=linestyles.get(ds, '-'), 
                                 marker=markers.get(ds, 'o'), 
                                 label=f'{ds} (train)'))
        plt.legend(handles=handles, title='Dataset (Set)', loc='best')
        
        # Save figure
        if SAVE_RESULTS:
            out_path = os.path.join(RESULTS_DIR, 'depth_sweep.pdf')
            plt.savefig(out_path, bbox_inches='tight', dpi=200)
            print(f'Phase A plot saved: {out_path}')
        plt.close()
        
    except Exception as e:
        print(f'Combined Phase A plotting failed: {e}')


def plot_phase_b_combined(dfs: List[pd.DataFrame]) -> None:
    """
    Plot max_features vs accuracy, colored by depth, across all datasets.
    """
    try:
        # Combine all datasets
        combined = pd.concat(dfs, ignore_index=True)
        
        # Ensure max_features is numeric (already done by find_max_features)
        combined = combined.copy()
        combined['depth_numeric'] = combined['max_depth'].astype(int)

        # Aggregate over seeds for robust estimates
        agg_funcs = {'test_accuracy_mean': ['mean', 'std']}
        aggregated = combined.groupby(['dataset', 'depth_numeric', 'max_features']).agg(agg_funcs).reset_index()
        aggregated.columns = ['dataset', 'depth_numeric', 'max_features', 
                             'test_accuracy_mean', 'test_accuracy_std']

        # Set up figure with discrete color mapping for depths
        fig, ax = plt.subplots(figsize=(10, 6))
        depths_sorted = sorted(aggregated['depth_numeric'].unique())
        discrete_colors = sns.color_palette('viridis', n_colors=len(depths_sorted))
        
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(discrete_colors)
        # Boundaries between depth colors (for discrete colorbar)
        boundaries = np.arange(-0.5, len(depths_sorted) + 0.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)
        depth_to_idx = {d: i for i, d in enumerate(depths_sorted)}
        
        # Get dataset-specific styling
        markers = _dataset_markers()
        linestyles = _dataset_linestyles()
        
        # Plot each (dataset, depth) combination
        for (ds, depth), sub in aggregated.groupby(['dataset', 'depth_numeric']):
            sub = sub.sort_values('max_features')
            color = cmap(depth_to_idx[depth])
            ax.plot(sub['max_features'], sub['test_accuracy_mean'], 
                   marker=markers.get(ds, 'o'), 
                   linestyle=linestyles.get(ds, '-'), 
                   color=color, linewidth=1.4, alpha=0.85, 
                   label='_nolegend_')
        
        ax.set_xlabel('Number of Features per Split')
        ax.set_ylabel('Average Validation Accuracy (%)')
        
        # Add colorbar for depth
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, boundaries=boundaries, 
                           ticks=range(len(depths_sorted)))
        cbar.set_label('Maximum Tree Depth')
        cbar.set_ticks(range(len(depths_sorted)))
        cbar.set_ticklabels([str(d) for d in depths_sorted])
        
        # Build dataset legend (placed above plot to avoid colorbar overlap)
        from matplotlib.lines import Line2D
        ds_unique = list(combined['dataset'].unique())
        handles = [Line2D([0], [0], color='black', 
                         linestyle=linestyles.get(ds, '-'), 
                         marker=markers.get(ds, 'o'), 
                         label=ds) for ds in ds_unique]
        ax.legend(handles=handles, bbox_to_anchor=(0.5, 1.15), 
                 loc='upper center', ncol=min(4, len(handles)), title='Dataset')
        
        # Save figure
        if SAVE_RESULTS:
            out_path = os.path.join(RESULTS_DIR, 'joint_feature_depth_sweep.pdf')
            fig.savefig(out_path, bbox_inches='tight', dpi=200)
            print(f'Phase B plot saved: {out_path}')
        plt.close(fig)
        
    except Exception as e:
        print(f'Combined Phase B plotting failed: {e}')


def plot_phase_c_combined(dfs: List[pd.DataFrame]) -> None:
    """
    Plot ensemble size vs accuracy, colored by depth, across all datasets.
    """
    try:
        # Combine all datasets
        combined = pd.concat(dfs, ignore_index=True)
        combined['depth_numeric'] = combined['max_depth'].astype(int)
        
        # Aggregate over seeds
        agg_funcs = {'test_accuracy_mean': ['mean', 'std']}
        aggregated = combined.groupby(['dataset', 'depth_numeric', 'n_estimators']).agg(agg_funcs).reset_index()
        aggregated.columns = ['dataset', 'depth_numeric', 'n_estimators', 
                             'test_accuracy_mean', 'test_accuracy_std']
        
        # Set up figure with discrete color mapping for depths
        fig, ax = plt.subplots(figsize=(10, 6))
        depths_sorted = sorted(aggregated['depth_numeric'].unique())
        discrete_colors = sns.color_palette('viridis', n_colors=len(depths_sorted))
        
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(discrete_colors)
        boundaries = np.arange(-0.5, len(depths_sorted) + 0.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)
        depth_to_idx = {d: i for i, d in enumerate(depths_sorted)}
        
        # Get dataset-specific styling
        markers = _dataset_markers()
        linestyles = _dataset_linestyles()
        
        # Plot each (dataset, depth) combination
        for (ds, depth), sub in aggregated.groupby(['dataset', 'depth_numeric']):
            sub = sub.sort_values('n_estimators')
            color = cmap(depth_to_idx[depth])
            ax.plot(sub['n_estimators'], sub['test_accuracy_mean'], 
                   marker=markers.get(ds, 'o'), 
                   linestyle=linestyles.get(ds, '-'), 
                   color=color, linewidth=1.6, alpha=0.9, 
                   label='_nolegend_')
        
        ax.set_xlabel('Number of Trees (Ensemble Size)')
        ax.set_ylabel('Average Validation Accuracy (%)')
        # Use log scale if range is large (helps visualise early improvements)
        ax.set_xscale('log' if aggregated['n_estimators'].max() > 50 else 'linear')
        
        # Add colorbar for depth
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, boundaries=boundaries, 
                           ticks=range(len(depths_sorted)))
        cbar.set_label('Maximum Tree Depth')
        cbar.set_ticks(range(len(depths_sorted)))
        cbar.set_ticklabels([str(d) for d in depths_sorted])
        
        # Build dataset legend
        from matplotlib.lines import Line2D
        ds_unique = list(combined['dataset'].unique())
        handles = [Line2D([0], [0], color='black', 
                         linestyle=linestyles.get(ds, '-'), 
                         marker=markers.get(ds, 'o'), 
                         label=ds) for ds in ds_unique]
        ax.legend(handles=handles, bbox_to_anchor=(0.5, 1.15), 
                 loc='upper center', ncol=min(4, len(handles)), title='Dataset')
        
        # Save figure
        if SAVE_RESULTS:
            out_path = os.path.join(RESULTS_DIR, 'joint_depth_n_estimators_sweep.pdf')
            fig.savefig(out_path, bbox_inches='tight', dpi=200)
            print(f'Phase C plot saved: {out_path}')
        plt.close(fig)
        
    except Exception as e:
        print(f'Combined Phase C plotting failed: {e}')


def plot_phase_d_combined(dfs: List[pd.DataFrame]) -> None:
    """
    Plot max_features vs accuracy at best depth across all datasets.
    """
    try:
        # Combine all datasets
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.copy()
        
        # Aggregate over seeds
        agg_funcs = {
            'test_accuracy_mean': ['mean', 'std'],
            'train_accuracy_mean': ['mean', 'std']
        }
        aggregated = combined.groupby(['dataset', 'max_features']).agg(agg_funcs).reset_index()
        aggregated.columns = ['dataset', 'max_features', 'test_accuracy_mean', 'test_accuracy_std', 
                              'train_accuracy_mean', 'train_accuracy_std']
        
        # Set up styling
        markers = _dataset_markers()
        linestyles = _dataset_linestyles()
        palette = sns.color_palette('colorblind')
        metric_colors = {'test': palette[0], 'train': palette[1]}  # Blue and orange
        
        plt.figure(figsize=(9, 6))
        
        # Plot each dataset
        for ds, sub in aggregated.groupby('dataset'):
            sub = sub.sort_values('max_features')
            
            # Plot test accuracy with error bars
            plt.errorbar(sub['max_features'], sub['test_accuracy_mean'], 
                        yerr=sub['test_accuracy_std'], 
                        fmt='none', ecolor=metric_colors['test'], alpha=0.5, 
                        capsize=3, label='_nolegend_')
            plt.plot(sub['max_features'], sub['test_accuracy_mean'], 
                    marker=markers.get(ds, 'o'), 
                    linestyle=linestyles.get(ds, '-'), 
                    color=metric_colors['test'], alpha=0.8, 
                    label='_nolegend_')
            
            # Plot training accuracy with error bars
            plt.errorbar(sub['max_features'], sub['train_accuracy_mean'], 
                        yerr=sub['train_accuracy_std'], 
                        fmt='none', ecolor=metric_colors['train'], alpha=0.5, 
                        capsize=3, label='_nolegend_')
            plt.plot(sub['max_features'], sub['train_accuracy_mean'], 
                    marker=markers.get(ds, 'o'), 
                    linestyle=linestyles.get(ds, '-'), 
                    color=metric_colors['train'], alpha=0.8, 
                    label='_nolegend_')
        
        plt.xlabel('Number of Features per Split')
        plt.ylabel('Average Accuracy (%)')
        plt.grid(True, alpha=0.4)
        
        # Build custom legend
        from matplotlib.lines import Line2D
        handles = []
        for ds in combined['dataset'].unique():
            handles.append(Line2D([0], [0], color=metric_colors['test'], 
                                 linestyle=linestyles.get(ds, '-'), 
                                 marker=markers.get(ds, 'o'), 
                                 label=f'{ds} (test)'))
            handles.append(Line2D([0], [0], color=metric_colors['train'], 
                                 linestyle=linestyles.get(ds, '-'), 
                                 marker=markers.get(ds, 'o'), 
                                 label=f'{ds} (train)'))
        plt.legend(handles=handles, title='Dataset (Set)', loc='best')
        
        # Save figure
        if SAVE_RESULTS:
            out_path = os.path.join(RESULTS_DIR, 'best_depth_feature_sweep.pdf')
            plt.savefig(out_path, bbox_inches='tight', dpi=200)
            print(f'Phase D plot saved: {out_path}')
        plt.close()
        
    except Exception as e:
        print(f'Combined Phase D plotting failed: {e}')