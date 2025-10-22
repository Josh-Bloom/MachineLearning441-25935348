"""
Utility functions for model evaluation and analysis.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Tuple
from config import SCORING, SEED, N_JOBS

def set_seed(seed: int = SEED) -> None:
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)


def evaluate_config(
    X: np.ndarray, 
    y: np.ndarray, 
    clf: RandomForestClassifier, 
    cv: StratifiedKFold,
    scoring: Dict[str, str] = SCORING, 
    n_jobs: int = N_JOBS
) -> Tuple[Dict[str, float], List[RandomForestClassifier]]:
    """
    Perform cross-validation and compute aggregated metrics.
    """
    # Run cross-validation with specified scoring metrics
    # return_train_score=True gives us both training and validation metrics
    # return_estimator=True preserves fitted models for analysis
    res = cross_validate(
        clf, X, y, 
        cv=cv, 
        scoring=scoring, 
        return_train_score=True, 
        n_jobs=n_jobs, 
        return_estimator=True
    )

    # Aggregate results across folds
    summary = {}
    for key in res:
        # Process test (validation) scores
        if key.startswith('test_'):
            # Convert accuracy to percentage for interpretability
            factor = 100 if key == 'test_accuracy' else 1
            summary[f'{key}_mean'] = np.mean(res[key]) * factor
            summary[f'{key}_std'] = np.std(res[key]) * factor
            
        # Process training scores
        elif key.startswith('train_'):
            # Convert accuracy to percentage for interpretability
            factor = 100 if key == 'train_accuracy' else 1
            summary[f'{key}_mean'] = np.mean(res[key]) * factor
            summary[f'{key}_std'] = np.std(res[key]) * factor
            
        # Process timing information
        elif key in ('fit_time', 'score_time'):
            summary[f'{key}_mean'] = np.mean(res[key])
            summary[f'{key}_std'] = np.std(res[key])
    
    return summary, res['estimator']


def find_max_depth(rf_list: List[RandomForestClassifier]) -> int:
    """
    Extract actual maximum depth from fitted Random Forests.
    """
    max_depth = 0
    for rf in rf_list:
        # Each rf.estimators_ is a list of decision trees
        # tree.tree_.max_depth gives the depth of that tree
        for tree in rf.estimators_:
            max_depth = max(max_depth, tree.tree_.max_depth)
    return max_depth


def find_max_features(mf: Any, num_features: int) -> int:
    """
    Convert max_features parameter to integer count.
    """
    if mf == 'sqrt':
        return int(np.sqrt(num_features))
    elif mf == 'log2':
        return int(np.log2(num_features))
    elif isinstance(mf, float):
        # For fractional values, multiply by total and ensure at least 1
        return max(1, int(num_features * mf))
    elif isinstance(mf, int):
        # For integer values, ensure at least 1
        return max(1, mf)
    else:
        # Fallback: use sqrt as default
        return int(np.sqrt(num_features))