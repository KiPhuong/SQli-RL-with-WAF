"""
State management for the SQL injection RL environment
Handles token-based state representation (200 tokens)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque


class StateManager:
    """
    Manages token-based state representation for the RL agent.
    State is represented as a sequence of 200 tokens (normalized payload).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state manager.
        
        Args:
            config: State configuration dictionary
        """
        self.config = config
        self.state_length = config.get('state_length', 200)  # Fixed token sequence length
        self.history_length = config.get('history_length', 10)
        
        # Current state (token sequence)
        self.current_state = [87] * self.state_length  # Initialize with PAD tokens (87)
        
        # History tracking
        self.state_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        
        # Environment context
        self.waf_detected = False
        self.injection_detected = False
        self.attempt_count = 0
        self.last_response = None
    
    def get_initial_state(self, waf_detected: bool = False, 
                         baseline_response: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Get the initial state of the environment.
        
        Args:
            waf_detected: Whether WAF was detected
            baseline_response: Baseline response for comparison
            
        Returns:
            Initial state as token sequence
        """
        # Reset state to all PAD tokens
        self.current_state = [87] * self.state_length  # PAD token = 87
        
        # Set environment context
        self.waf_detected = waf_detected
        self.injection_detected = False
        self.attempt_count = 0
        self.last_response = baseline_response
        
        # Clear history
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        return self.current_state.copy()
    
    def get_state(self) -> List[int]:
        """
        Get the current state as token sequence.
        
        Returns:
            Current state token sequence
        """
        return self.current_state.copy()
    
    def get_state_size(self) -> int:
        """
        Get the state size (number of tokens).
        
        Returns:
            State size
        """
        return self.state_length
    
    def update_state(self, new_token: int, response: Dict[str, Any], 
                    analysis: Dict[str, Any], attempt_count: int) -> List[int]:
        """
        Update the state by adding a new token.
        
        Args:
            new_token: New token to add to state
            response: Response received
            analysis: Response analysis
            attempt_count: Current attempt count
            
        Returns:
            Updated state token sequence
        """
        # Add new token to state
        self.current_state = self._add_token_to_state(new_token)
        
        # Update environment context
        self.attempt_count = attempt_count
        self.last_response = response
        self.injection_detected = analysis.get('injection_detected', False)
        
        # Update history
        self.state_history.append(self.current_state.copy())
        self.action_history.append(new_token)
        
        return self.current_state.copy()
    
    def _add_token_to_state(self, new_token: int) -> List[int]:
        """
        Add a new token to the current state.
        
        Args:
            new_token: Token to add
            
        Returns:
            Updated state
        """
        # Find the first PAD token and replace it
        for i, token in enumerate(self.current_state):
            if token == 87:  # PAD token
                self.current_state[i] = new_token
                break
        else:
            # If no PAD token found, shift left and add at end
            self.current_state = self.current_state[1:] + [new_token]
        
        return self.current_state
    
    def get_current_payload_length(self) -> int:
        """
        Get the current payload length (excluding PAD tokens).
        
        Returns:
            Number of non-PAD tokens
        """
        return len([token for token in self.current_state if token != 87])
    
    def is_state_full(self) -> bool:
        """
        Check if the state is full (no PAD tokens).
        
        Returns:
            True if state is full
        """
        return 87 not in self.current_state
    
    def reset_state(self):
        """Reset the state to initial empty state."""
        self.current_state = [87] * self.state_length
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.attempt_count = 0
        self.injection_detected = False
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state as numpy array for neural network input.
        
        Returns:
            State as numpy array
        """
        return np.array(self.current_state, dtype=np.float32)
    
    def get_context_info(self) -> Dict[str, Any]:
        """
        Get additional context information.
        
        Returns:
            Context information dictionary
        """
        return {
            'waf_detected': self.waf_detected,
            'injection_detected': self.injection_detected,
            'attempt_count': self.attempt_count,
            'payload_length': self.get_current_payload_length(),
            'state_full': self.is_state_full()
        }
