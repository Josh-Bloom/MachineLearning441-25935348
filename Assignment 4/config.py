"""
Configuration parameters for Random Forest experiments.
"""

from typing import Dict, List, Optional, Union

# Random seed for reproducibility
SEED: int = 0

# List of seeds for multi-seed experiments
SEEDS: List[int] = [SEED]

# Scoring metrics for cross-validation
SCORING: Dict[str, str] = {
    'accuracy': 'balanced_accuracy',
    'f1_macro': 'f1_macro'
}

# Number of parallel jobs for sklearn operations
N_JOBS: int = 2

# Number of cross-validation folds
CV_FOLDS: int = 4

# Output directory for results
RESULTS_DIR: str = 'results'

# Enable/disable saving results
SAVE_RESULTS: bool = True

# Default number of trees in forest
DEFAULT_N_ESTIMATORS: int = 300

# Default max_features parameter
DEFAULT_MAX_FEATURES: Union[str, float, int] = 'sqrt'

# Tree depth values for Phase A
DEFAULT_DEPTH_VALUES: List[Optional[int]] = [1, 2, 3, 4, 6, 8, 10, 15, 20, None]

# Reduced depth range for Phases B, C
DEFAULT_DEPTH_VALUES_SLIM: List[Optional[int]] = [1, 6, 15, None]

# Max_features values to test in Phases B, D
DEFAULT_MAX_FEATURES_VALUES: List[Union[str, float, int]] = ['sqrt', 'log2', 1.0, 0.75, 0.5, 0.25, 1]

# Ensemble sizes for Phase C
DEFAULT_N_ESTIMATORS_VALUES: List[int] = [1, 5, 10, 50, 100, 300, 600]