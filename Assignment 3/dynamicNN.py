import torch
import torch.nn as nn
import copy

class DynamicNetwork(nn.Module):
    """Dynamic neural network with growth control.
    
    mode = "reinit"   → reinitialize all weights on growth
    mode = "preserve" → copy old weights, init only new ones
    """
    
    def __init__(self, input_size, num_classes=2, hidden_neurons=0, mode="reinit"):
        super().__init__()
        assert mode in ["reinit", "preserve"], "mode must be 'reinit' or 'preserve'"
        
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.num_classes = num_classes
        self.mode = mode
        self.activation = nn.ReLU()
        self._build_model()
        self.best_state = None
        self.best_metric = None
    
    def _build_model(self):
        """(Re)build network with current size"""
        if self.hidden_neurons == 0:
            self.hidden_layer = None
            self.output_layer = nn.Linear(self.input_size, self.num_classes)
            nn.init.xavier_uniform_(self.output_layer.weight)
        else:
            self.hidden_layer = nn.Linear(self.input_size, self.hidden_neurons)
            self.output_layer = nn.Linear(self.hidden_neurons, self.num_classes)
            nn.init.kaiming_normal_(self.hidden_layer.weight, nonlinearity="relu")
            nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        if self.hidden_layer is None:
            return self.output_layer(x)
        else:
            hidden_output = self.activation(self.hidden_layer(x))
            return self.output_layer(hidden_output)
    
    def add_hidden_neuron(self, k=1):
        """Expand hidden neurons"""
        if self.mode == "reinit":
            self.hidden_neurons += k
            self._build_model()
            return
        
        # --- preserve mode ---
        old_hidden = self.hidden_layer
        old_output = self.output_layer
        new_hidden_neurons = self.hidden_neurons + k
        
        if self.hidden_neurons == 0:
            new_hidden = nn.Linear(self.input_size, new_hidden_neurons)
            new_output = nn.Linear(new_hidden_neurons, self.num_classes)
            nn.init.kaiming_normal_(new_hidden.weight, nonlinearity="relu")
            nn.init.xavier_uniform_(new_output.weight)
        
        else:
            new_hidden = nn.Linear(self.input_size, new_hidden_neurons)
            new_output = nn.Linear(new_hidden_neurons, self.num_classes)
            
            with torch.no_grad():
                # copy old hidden
                new_hidden.weight[:self.hidden_neurons, :] = old_hidden.weight.data.clone()
                new_hidden.bias[:self.hidden_neurons] = old_hidden.bias.data.clone()
                # init new
                nn.init.kaiming_normal_(new_hidden.weight[self.hidden_neurons:], nonlinearity="relu")
                nn.init.zeros_(new_hidden.bias[self.hidden_neurons:])
                
                # copy old output
                new_output.weight[:, :self.hidden_neurons] = old_output.weight.data.clone()
                new_output.bias.data = old_output.bias.data.clone()
                # init new output connections
                nn.init.xavier_uniform_(new_output.weight[:, self.hidden_neurons:])
        
        self.hidden_layer = new_hidden
        self.output_layer = new_output
        self.hidden_neurons = new_hidden_neurons
    
    def add_output_class(self, k=1):
        """Expand output classes"""
        if self.mode == "reinit":
            self.num_classes += k
            self._build_model()
            return
        
        # --- preserve mode ---
        old_output = self.output_layer
        new_num_classes = self.num_classes + k
        
        if self.hidden_neurons == 0:
            new_output = nn.Linear(self.input_size, new_num_classes)
        else:
            new_output = nn.Linear(self.hidden_neurons, new_num_classes)
        
        with torch.no_grad():
            # copy old
            new_output.weight[:self.num_classes, :] = old_output.weight.data.clone()
            new_output.bias[:self.num_classes] = old_output.bias.data.clone()
            # init new
            nn.init.xavier_uniform_(new_output.weight[self.num_classes:, :])
            nn.init.zeros_(new_output.bias[self.num_classes:])
        
        self.output_layer = new_output
        self.num_classes = new_num_classes

    # def store(self, metric):
    #     if self.best_metric is None or metric < self.best_metric:
    #         self.best_state = copy.deepcopy(self.state_dict())
    #         self.best_metric = metric

    # def restore(self):
    #     if self.best_state is not None:
    #         self.load_state_dict(self.best_state)
    
    def store_architecture_checkpoint(self, loss, accuracy, hidden_neurons):
        """Store a checkpoint for a specific architecture"""
        if not hasattr(self, 'architecture_checkpoints'):
            self.architecture_checkpoints = {}
        
        # Store checkpoint for this architecture
        self.architecture_checkpoints[hidden_neurons] = {
            'state_dict': copy.deepcopy(self.state_dict()),
            'loss': loss,
            'accuracy': accuracy,
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }
        
        # Update best overall architecture using a combined metric
        if not hasattr(self, 'best_architecture_loss'):
            # First architecture - always store
            self.best_architecture_loss = loss
            self.best_architecture_accuracy = accuracy
            self.best_architecture_neurons = hidden_neurons
        else:
            # Use a combined score: prioritize lower loss, but also consider accuracy
            # Score = -loss + accuracy (higher is better)
            ACC_WEIGHT = 3
            current_score = -loss + ACC_WEIGHT * accuracy
            best_score = -self.best_architecture_loss + ACC_WEIGHT * self.best_architecture_accuracy
            
            if current_score > best_score:
                self.best_architecture_loss = loss
                self.best_architecture_accuracy = accuracy
                self.best_architecture_neurons = hidden_neurons
    
    def restore_architecture_checkpoint(self, hidden_neurons):
        """Restore to a specific architecture checkpoint"""
        if not hasattr(self, 'architecture_checkpoints'):
            return False
            
        if hidden_neurons not in self.architecture_checkpoints:
            return False
        
        checkpoint = self.architecture_checkpoints[hidden_neurons]
        
        # Restore the architecture
        self.hidden_neurons = hidden_neurons
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        
        # Rebuild the model with the correct architecture
        self._build_model()
        
        # Load the saved state
        self.load_state_dict(checkpoint['state_dict'])
        
        return True
    
    def restore_best_architecture(self):
        """Restore to the best architecture found across all attempts"""
        if hasattr(self, 'best_architecture_neurons'):
            return self.restore_architecture_checkpoint(self.best_architecture_neurons)
        return False
    
    def list_checkpoints(self):
        """List all available architecture checkpoints"""
        if not hasattr(self, 'architecture_checkpoints'):
            return {}
        return {neurons: {'loss': checkpoint['loss'], 'accuracy': checkpoint['accuracy']} 
                for neurons, checkpoint in self.architecture_checkpoints.items()}
    