"""
Reinforcement Learning Agent for SQL Injection Penetration Testing
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from .neural_network import DQN
from .boltzmann_exploration import BoltzmannExploration
from .action_space import ActionSpace


class RLAgent:
    """
    Deep Q-Network based RL agent for automated SQL injection testing
    with WAF bypass capabilities.
    """
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        """
        Initialize the RL agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            config: Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Neural network
        self.q_network = DQN(state_size, action_size, config.get('hidden_layers', [256, 128]))
        self.target_q_network = DQN(state_size, action_size, config.get('hidden_layers', [256, 128]))
        
        # Exploration strategy
        self.exploration = BoltzmannExploration(
            initial_temperature=config.get('initial_temperature', 1.0),
            min_temperature=config.get('min_temperature', 0.1),
            decay_rate=config.get('temperature_decay', 0.995)
        )
        
        # Action space
        self.action_space = ActionSpace()
        
        # Training parameters
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        self.memory = []
        self.memory_size = config.get('memory_size', 10000)
        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.99)
        self.update_target_frequency = config.get('update_target_frequency', 100)
        self.step_count = 0
        
    def act(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        
        # Use Boltzmann exploration
        action = self.exploration.select_action(q_values.detach().numpy()[0])
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append(experience)
    
    def replay(self):
        """
        Train the agent using experiences from replay memory.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]
        
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences])
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences])
        
        # Calculate target Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.update_target_network()
        
        # Update exploration
        self.exploration.update()
    
    def update_target_network(self):
        """
        Update target network with current network weights.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
        """
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
