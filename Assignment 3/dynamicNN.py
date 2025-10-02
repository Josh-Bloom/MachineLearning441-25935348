"""
Dynamic Neural Network implementation for incremental learning.

This module provides a DynamicNetwork class that can grow its architecture during training
by adding hidden neurons and output classes while preserving learned weights.

Key Features:
- Dynamic architecture growth (add hidden neurons and output classes)
- Weight preservation when growing the network
- Architecture checkpointing and restoration
- Support for both 'preserve' and 'reinit' modes
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DynamicNetwork(nn.Module):
    """
    Dynamic neural network with architecture growth capabilities for incremental learning.
    """
    
    def __init__(self, input_size: int, num_classes: int = 2, hidden_neurons: int = 0, mode: str = "reinit"):
        """
        Initialise the dynamic neural network.
        """
        super().__init__()
        assert mode in ["reinit", "preserve"], "mode must be 'reinit' or 'preserve'"
        
        # Store network parameters
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.num_classes = num_classes
        self.mode = mode
        self.activation = nn.ReLU()
        
        # Architecture tracking for checkpointing
        self.architecture_checkpoints: Dict[int, Dict[str, Any]] = {}
        self.best_architecture_f1_score: Optional[float] = None
        self.best_architecture_neurons: Optional[int] = None
        
        # Build the initial network architecture
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Build or rebuild the network architecture with current parameters.
        """
        if self.hidden_neurons == 0:
            # Direct input to output mapping (no hidden layer)
            self.hidden_layer = None
            self.output_layer = nn.Linear(self.input_size, self.num_classes)
            # Glorot uniform initialisation for output layer
            nn.init.xavier_uniform_(self.output_layer.weight)
        else:
            # Input -> Hidden -> Output mapping
            self.hidden_layer = nn.Linear(self.input_size, self.hidden_neurons)
            self.output_layer = nn.Linear(self.hidden_neurons, self.num_classes)
            # He normal initialisation for hidden layer (good for ReLU)
            nn.init.kaiming_normal_(self.hidden_layer.weight, nonlinearity="relu")
            # Glorot uniform initialisation for output layer
            nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        if self.hidden_layer is None:
            # Direct input to output mapping
            return self.output_layer(x)
        else:
            # Input -> Hidden -> Output mapping
            hidden_output = self.activation(self.hidden_layer(x))
            return self.output_layer(hidden_output)
    
    def add_hidden_neuron(self, k: int = 1) -> None:
        """
        Add k hidden neurons to the network.
        """
        if self.mode == "reinit":
            # Reinit mode: rebuild entire network with new random weights
            self.hidden_neurons += k
            self._build_model()
            return
        
        # Preserve mode: copy existing weights and initialise new ones
        self._add_hidden_neurons_preserve_mode(k)
    
    def _add_hidden_neurons_preserve_mode(self, k: int) -> None:
        """
        Add hidden neurons while preserving existing weights.
        """
        old_hidden = self.hidden_layer
        old_output = self.output_layer
        new_hidden_neurons = self.hidden_neurons + k
        
        if self.hidden_neurons == 0:
            # First hidden layer: create new layers and initialise weights
            new_hidden = nn.Linear(self.input_size, new_hidden_neurons)
            new_output = nn.Linear(new_hidden_neurons, self.num_classes)
            nn.init.kaiming_normal_(new_hidden.weight, nonlinearity="relu")
            nn.init.xavier_uniform_(new_output.weight)
        else:
            # Expand existing hidden layer: preserve existing weights
            new_hidden = nn.Linear(self.input_size, new_hidden_neurons)
            new_output = nn.Linear(new_hidden_neurons, self.num_classes)
            
            with torch.no_grad():
                # Copy existing hidden layer weights
                new_hidden.weight[:self.hidden_neurons, :] = old_hidden.weight.data.clone()
                new_hidden.bias[:self.hidden_neurons] = old_hidden.bias.data.clone()
                
                # Initialise new hidden layer weights
                nn.init.kaiming_normal_(new_hidden.weight[self.hidden_neurons:], nonlinearity="relu")
                nn.init.zeros_(new_hidden.bias[self.hidden_neurons:])
                
                # Copy existing output layer weights
                new_output.weight[:, :self.hidden_neurons] = old_output.weight.data.clone()
                new_output.bias.data = old_output.bias.data.clone()
                
                # Initialise new output connections
                nn.init.xavier_uniform_(new_output.weight[:, self.hidden_neurons:])
        
        # Update network layers and neuron count
        self.hidden_layer = new_hidden
        self.output_layer = new_output
        self.hidden_neurons = new_hidden_neurons
    
    def add_output_class(self, k: int = 1) -> None:
        """
        Add k output classes to the network.
        """
        if self.mode == "reinit":
            # Reinit mode: rebuild entire network with new random weights
            self.num_classes += k
            self._build_model()
            return
        
        # Preserve mode: copy existing weights and initialise new ones
        self._add_output_classes_preserve_mode(k)
    
    def _add_output_classes_preserve_mode(self, k: int) -> None:
        """
        Add output classes while preserving existing weights.
        """
        old_output = self.output_layer
        new_num_classes = self.num_classes + k
        
        # Create new output layer with expanded size
        if self.hidden_neurons == 0:
            new_output = nn.Linear(self.input_size, new_num_classes)
        else:
            new_output = nn.Linear(self.hidden_neurons, new_num_classes)
        
        with torch.no_grad():
            # Copy existing output weights
            new_output.weight[:self.num_classes, :] = old_output.weight.data.clone()
            new_output.bias[:self.num_classes] = old_output.bias.data.clone()
            
            # Initialise new output weights
            nn.init.xavier_uniform_(new_output.weight[self.num_classes:, :])
            nn.init.zeros_(new_output.bias[self.num_classes:])
        
        # Update output layer and class count
        self.output_layer = new_output
        self.num_classes = new_num_classes
    
    def store_architecture_checkpoint(self, f1_score: float, hidden_neurons: int) -> None:
        """
        Store a checkpoint for a specific architecture.
        """
        # Store checkpoint for this architecture
        self.architecture_checkpoints[hidden_neurons] = {
            'state_dict': copy.deepcopy(self.state_dict()),
            'f1_score': f1_score,
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }
        
        # Update best overall architecture using F1 score
        self._update_best_architecture(f1_score, hidden_neurons)
    
    def _update_best_architecture(self, f1_score: float, hidden_neurons: int) -> None:
        """
        Update the best architecture tracking.
        """
        if self.best_architecture_f1_score is None:
            # First architecture - always store as best
            self.best_architecture_f1_score = f1_score
            self.best_architecture_neurons = hidden_neurons
        else:
            # Update if current architecture is better
            if f1_score > self.best_architecture_f1_score:
                self.best_architecture_f1_score = f1_score
                self.best_architecture_neurons = hidden_neurons
    
    def restore_architecture_checkpoint(self, hidden_neurons: int) -> bool:
        """
        Restore to a specific architecture checkpoint.
        """
        if hidden_neurons not in self.architecture_checkpoints:
            logger.warning(f"No checkpoint found for {hidden_neurons} hidden neurons")
            return False
        
        checkpoint = self.architecture_checkpoints[hidden_neurons]
        
        # Restore the architecture parameters
        self.hidden_neurons = hidden_neurons
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        
        # Rebuild the model with the correct architecture
        self._build_model()
        
        # Load the saved weights
        self.load_state_dict(checkpoint['state_dict'])
        
        return True
    
    def restore_best_architecture(self) -> bool:
        """
        Restore to the best architecture found across all attempts.
        """
        if self.best_architecture_neurons is not None:
            return self.restore_architecture_checkpoint(self.best_architecture_neurons)
        return False