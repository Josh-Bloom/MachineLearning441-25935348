"""
Configuration management for incremental learning experiments.

This module provides centralised configuration management for all experiment parameters,
making it easy to modify settings and maintain consistency across different runs.

The configuration includes parameters for:
- Training (learning rate, epochs, batch size)
- Architecture search (max hidden neurons, patience)
- Model behavior (preserve vs reinit mode)
- Cross-validation (number of folds, test size)
"""

from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters in incremental learning experiments.
    """
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Data loading parameters
    batch_size: int = 32
    
    # Learning parameters
    learning_rate: float = 0.002
    fine_learning_rate: float = 0.0002  # Lower learning rate for fine-tuning neuron addition (preserve mode only)
    weight_decay: float = 1e-5
    
    # Training epochs
    max_epochs: int = 100
    fine_max_epochs: int = 60  # Fewer epochs for fine-tuning neuron addition (preserve mode only)
    
    # Early stopping parameters
    patience: int = 8
    fine_patience: int = 5  # Smaller patience window for fine-tuning neuron addition (preserve mode only)
    
    # Architecture search parameters
    max_hidden_neurons: int = 32
    
    # Model mode ('reinit' or 'preserve')
    model_mode: str = 'preserve'
    
    # Logging and verbosity
    verbose: bool = False
    
    # Cross-validation parameters
    n_folds: int = 4
    test_size: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        """
        return {
            'seed': self.seed,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'fine_learning_rate': self.fine_learning_rate,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'fine_max_epochs': self.fine_max_epochs,
            'patience': self.patience,
            'fine_patience': self.fine_patience,
            'max_hidden_neurons': self.max_hidden_neurons,
            'model_mode': self.model_mode,
            'verbose': self.verbose,
            'n_folds': self.n_folds,
            'test_size': self.test_size
        }

# Predefined configurations for different experiment types
DEFAULT_CONFIG = TrainingConfig()