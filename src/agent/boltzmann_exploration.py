"""
Boltzmann exploration strategy for RL agent
"""

import numpy as np
from typing import Optional


class BoltzmannExploration:
    """
    Boltzmann (softmax) exploration strategy for action selection.
    Uses temperature-based probability distribution for balanced exploration/exploitation.
    """
    
    def __init__(self, initial_temperature: float = 1.0, min_temperature: float = 0.1, 
                 decay_rate: float = 0.995):
        """
        Initialize Boltzmann exploration.
        
        Args:
            initial_temperature: Starting temperature for exploration
            min_temperature: Minimum temperature threshold
            decay_rate: Rate at which temperature decays
        """
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.step_count = 0
    
    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select action using Boltzmann distribution.
        
        Args:
            q_values: Q-values for all possible actions
            
        Returns:
            Selected action index
        """
        if self.temperature <= 0:
            # Greedy selection when temperature is zero
            return np.argmax(q_values)
        
        # Apply temperature scaling
        scaled_q_values = q_values / self.temperature
        
        # Prevent overflow by subtracting max value
        scaled_q_values = scaled_q_values - np.max(scaled_q_values)
        
        # Compute softmax probabilities
        exp_values = np.exp(scaled_q_values)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action based on probabilities
        action = np.random.choice(len(q_values), p=probabilities)
        return action
    
    def get_action_probabilities(self, q_values: np.ndarray) -> np.ndarray:
        """
        Get action selection probabilities without sampling.
        
        Args:
            q_values: Q-values for all possible actions
            
        Returns:
            Probability distribution over actions
        """
        if self.temperature <= 0:
            # One-hot distribution for greedy selection
            probabilities = np.zeros_like(q_values)
            probabilities[np.argmax(q_values)] = 1.0
            return probabilities
        
        # Apply temperature scaling
        scaled_q_values = q_values / self.temperature
        
        # Prevent overflow by subtracting max value
        scaled_q_values = scaled_q_values - np.max(scaled_q_values)
        
        # Compute softmax probabilities
        exp_values = np.exp(scaled_q_values)
        probabilities = exp_values / np.sum(exp_values)
        
        return probabilities
    
    def update(self):
        """
        Update temperature based on decay schedule.
        """
        self.step_count += 1
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.decay_rate
        )
    
    def reset_temperature(self):
        """
        Reset temperature to initial value.
        """
        self.temperature = self.initial_temperature
        self.step_count = 0
    
    def set_temperature(self, temperature: float):
        """
        Manually set temperature value.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = max(0.0, temperature)
    
    def get_temperature(self) -> float:
        """
        Get current temperature value.
        
        Returns:
            Current temperature
        """
        return self.temperature


class AdaptiveBoltzmannExploration(BoltzmannExploration):
    """
    Adaptive Boltzmann exploration that adjusts temperature based on performance.
    """
    
    def __init__(self, initial_temperature: float = 1.0, min_temperature: float = 0.1,
                 decay_rate: float = 0.995, adaptation_rate: float = 0.01):
        """
        Initialize adaptive Boltzmann exploration.
        
        Args:
            initial_temperature: Starting temperature for exploration
            min_temperature: Minimum temperature threshold
            decay_rate: Rate at which temperature decays
            adaptation_rate: Rate of adaptation based on performance
        """
        super().__init__(initial_temperature, min_temperature, decay_rate)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.performance_window = 100
    
    def update_with_performance(self, reward: float):
        """
        Update temperature based on recent performance.
        
        Args:
            reward: Recent reward signal
        """
        # Store performance
        self.performance_history.append(reward)
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        # Calculate recent performance
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            overall_performance = np.mean(self.performance_history)
            
            # Adapt temperature based on performance
            if recent_performance < overall_performance:
                # Increase exploration if performance is declining
                self.temperature = min(
                    self.initial_temperature,
                    self.temperature * (1 + self.adaptation_rate)
                )
            else:
                # Decrease exploration if performance is improving
                self.temperature = max(
                    self.min_temperature,
                    self.temperature * (1 - self.adaptation_rate)
                )
        
        # Apply regular decay
        super().update()


class EpsilonBoltzmannExploration:
    """
    Hybrid exploration combining epsilon-greedy and Boltzmann strategies.
    """
    
    def __init__(self, epsilon: float = 0.1, temperature: float = 1.0,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize epsilon-Boltzmann exploration.
        
        Args:
            epsilon: Probability of random action selection
            temperature: Temperature for Boltzmann selection
            epsilon_decay: Rate at which epsilon decays
            min_epsilon: Minimum epsilon value
        """
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        self.boltzmann = BoltzmannExploration(
            initial_temperature=temperature,
            min_temperature=0.1,
            decay_rate=0.999
        )
    
    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select action using hybrid epsilon-Boltzmann strategy.
        
        Args:
            q_values: Q-values for all possible actions
            
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            # Random action
            return np.random.randint(len(q_values))
        else:
            # Boltzmann action selection
            return self.boltzmann.select_action(q_values)
    
    def update(self):
        """
        Update both epsilon and temperature.
        """
        # Update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Update Boltzmann temperature
        self.boltzmann.update()
    
    def reset(self):
        """
        Reset exploration parameters.
        """
        self.epsilon = self.initial_epsilon
        self.boltzmann.reset_temperature()
