"""Neural network architectures for Module 06."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SimpleMLP(nn.Module):
    """
    Simple multi-layer perceptron for classification.
    
    Architecture:
        Input → [Linear → ReLU → Dropout]* → Linear → Output
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of classes)
        dropout: Dropout probability (applied after each hidden layer)
    
    Example:
        >>> model = SimpleMLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)
        >>> assert logits.shape == (32, 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation - will use with CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        # Flatten if needed (e.g., for MNIST images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)


class CNNMnist(nn.Module):
    """
    Convolutional neural network for MNIST/FashionMNIST.
    
    Architecture (LeNet-style):
        Conv(1→32, 3x3) → ReLU → MaxPool(2x2)
        Conv(32→64, 3x3) → ReLU → MaxPool(2x2)
        Flatten
        Linear(1600→128) → ReLU → Dropout
        Linear(128→10)
    
    Args:
        num_classes: Number of output classes (default: 10 for MNIST)
        dropout: Dropout probability after fully connected layer
    
    Example:
        >>> model = CNNMnist(num_classes=10, dropout=0.5)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
        >>> assert logits.shape == (32, 10)
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After two 2x2 pooling: 28x28 → 14x14 → 7x7
        # So feature map is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (N, 3136)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x


def create_model(config) -> nn.Module:
    """
    Factory function to create model from config.
    
    Args:
        config: TrainingConfig instance
    
    Returns:
        Instantiated model
    
    Example:
        >>> from config import TrainingConfig
        >>> config = TrainingConfig(model_type="SimpleMLP")
        >>> model = create_model(config)
    """
    if config.model_type == "SimpleMLP":
        return SimpleMLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )
    elif config.model_type == "CNNMnist":
        return CNNMnist(
            num_classes=config.output_dim,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
