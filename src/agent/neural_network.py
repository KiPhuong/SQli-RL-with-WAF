"""
Deep Q-Network implementation for SQL injection RL agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):
    """
    Deep Q-Network for value function approximation in SQL injection testing.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128]):
        """
        Initialize the DQN.
        
        Args:
            state_size: Size of the input state vector
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build the network layers
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize network weights using Xavier initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for all actions
        """
        return self.network(state)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture for improved value estimation.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128]):
        """
        Initialize the Dueling DQN.
        
        Args:
            state_size: Size of the input state vector
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        feature_layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers[:-1]:
            feature_layers.append(nn.Linear(input_size, hidden_size))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        self.feature_network = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize network weights using Xavier initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for all actions
        """
        features = self.feature_network(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class NoiseInjectionLayer(nn.Module):
    """
    Layer for injecting noise during training to improve robustness.
    """
    
    def __init__(self, noise_std: float = 0.1):
        """
        Initialize noise injection layer.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
        """
        super(NoiseInjectionLayer, self).__init__()
        self.noise_std = noise_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional noise injection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with noise (during training only)
        """
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class AttentionDQN(nn.Module):
    """
    DQN with attention mechanism for focusing on important state features.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128]):
        """
        Initialize the Attention DQN.
        
        Args:
            state_size: Size of the input state vector
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
        """
        super(AttentionDQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[0] // 2, hidden_layers[0]),
            nn.Softmax(dim=1)
        )
        
        # Value network
        value_layers = []
        input_size = hidden_layers[0]
        
        for hidden_size in hidden_layers[1:]:
            value_layers.append(nn.Linear(input_size, hidden_size))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        value_layers.append(nn.Linear(input_size, action_size))
        self.value_network = nn.Sequential(*value_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize network weights using Xavier initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for all actions
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Compute Q-values
        q_values = self.value_network(attended_features)
        
        return q_values
