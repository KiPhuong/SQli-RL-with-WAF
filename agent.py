"""
RL Agent for SQL Injection Testing
Uses DQN with Boltzmann exploration for token selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any


class DQN(nn.Module):
    """Deep Q-Network for token selection"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
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
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)


class BoltzmannExploration:
    """Boltzmann (Softmax) exploration strategy"""
    
    def __init__(self, initial_temperature: float = 2.0, min_temperature: float = 0.1, 
                 decay_rate: float = 0.99):
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
    
    def select_action(self, q_values: np.ndarray) -> int:
        """Select action using Boltzmann distribution"""
        # Apply temperature scaling
        scaled_q_values = q_values / max(self.temperature, self.min_temperature)
        
        # Compute softmax probabilities
        exp_values = np.exp(scaled_q_values - np.max(scaled_q_values))  # Numerical stability
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action based on probabilities
        action = np.random.choice(len(q_values), p=probabilities)
        return action
    
    def update_temperature(self):
        """Decay temperature over time"""
        self.temperature = max(self.min_temperature, self.temperature * self.decay_rate)
    
    def reset_temperature(self):
        """Reset temperature to initial value"""
        self.temperature = self.initial_temperature


class SQLiRLAgent:
    """RL Agent for SQL Injection Token Selection"""
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any] = None):
        self.state_size = state_size
        self.action_size = action_size

        # Default configuration with dynamic sizing
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'hidden_sizes': self._calculate_hidden_sizes(state_size, action_size),
            'initial_temperature': 2.0,
            'min_temperature': 0.1,
            'temperature_decay': 0.99999999
        }

        print(f"🧠 Neural Network Architecture:")
        print(f"   Input: {state_size} → Hidden: {default_config['hidden_sizes']} → Output: {action_size}")
        
        self.config = {**default_config, **(config or {})}
        
        # Neural networks
        self.q_network = DQN(state_size, action_size, self.config['hidden_sizes'])
        self.target_network = DQN(state_size, action_size, self.config['hidden_sizes'])
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
        
        # Exploration strategy
        self.exploration = BoltzmannExploration(
            self.config['initial_temperature'],
            self.config['min_temperature'],
            self.config['temperature_decay']
        )
        
        # Experience replay
        self.memory = deque(maxlen=self.config['memory_size'])
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()

    def _calculate_hidden_sizes(self, state_size: int, action_size: int) -> List[int]:
        """Calculate optimal hidden layer sizes based on input/output dimensions"""
        # Rule of thumb: hidden layers should be between input and output size
        max_hidden = max(state_size, action_size)
        min_hidden = min(state_size, action_size)

        if action_size <= 100:
            # Small action space
            return [512, 256, 128]
        elif action_size <= 500:
            # Medium action space
            return [1024, 512, 256]
        else:
            # Large action space
            return [2048, 1024, 512]
    
    def select_token(self, state: np.ndarray) -> int:
        """Select next token using Q-network and Boltzmann exploration"""
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # Select action using Boltzmann exploration
        action = self.exploration.select_action(q_values)
        return action
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given current state"""
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        return q_values
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the Q-network using experience replay"""
        if len(self.memory) < self.config['batch_size']:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config['target_update_freq'] == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_temperature(self):
        """Decay temperature after each episode"""
        self.exploration.update_temperature()
        #print(f"🌡️ Temperature decayed to: {self.exploration.temperature:.6f}")

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
    
    def get_exploration_info(self) -> Dict[str, float]:
        """Get current exploration parameters"""
        return {
            'temperature': self.exploration.temperature,
            'min_temperature': self.exploration.min_temperature,
            'decay_rate': self.exploration.decay_rate
        }
