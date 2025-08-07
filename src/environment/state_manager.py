"""
State management for the penetration testing environment
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import hashlib
import re


class StateManager:
    """
    Manages the state representation for the RL agent.
    Converts environment observations into numerical state vectors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state manager.
        
        Args:
            config: State configuration dictionary
        """
        self.config = config
        self.state_size = config.get('state_size', 100)
        self.history_length = config.get('history_length', 10)
        
        # State components
        self.current_payload_features = np.zeros(20)
        self.response_features = np.zeros(15)
        self.waf_features = np.zeros(10)
        self.injection_features = np.zeros(15)
        self.context_features = np.zeros(10)
        self.history_features = np.zeros(30)
        
        # History tracking
        self.payload_history = deque(maxlen=self.history_length)
        self.response_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        
        # Feature extractors
        self.payload_analyzer = PayloadAnalyzer()
        self.response_analyzer = ResponseAnalyzer()
        
        # Current state
        self.state = np.zeros(self.state_size)
    
    def get_initial_state(self, waf_detected: bool = False, 
                         baseline_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the initial state of the environment.
        
        Args:
            waf_detected: Whether WAF was detected
            baseline_response: Baseline response for comparison
            
        Returns:
            Initial state dictionary
        """
        # Reset all features
        self.current_payload_features = np.zeros(20)
        self.response_features = np.zeros(15)
        self.waf_features = np.zeros(10)
        self.injection_features = np.zeros(15)
        self.context_features = np.zeros(10)
        self.history_features = np.zeros(30)
        
        # Set initial WAF features
        if waf_detected:
            self.waf_features[0] = 1.0  # WAF detected flag
        
        # Set baseline response features
        if baseline_response:
            self.response_features[0] = min(baseline_response.get('status_code', 200) / 500.0, 1.0)
            self.response_features[1] = min(len(baseline_response.get('content', '')) / 10000.0, 1.0)
            self.response_features[2] = min(baseline_response.get('response_time', 0) / 10.0, 1.0)
        
        # Clear history
        self.payload_history.clear()
        self.response_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        # Build initial state vector
        self._build_state_vector()
        
        return {
            'waf_detected': waf_detected,
            'baseline_response': baseline_response,
            'attempt_count': 0,
            'injection_detected': False,
            'current_payload': '',
            'last_response': None
        }
    
    def update_state(self, action: int, payload: str, response: Dict[str, Any], 
                    analysis: Dict[str, Any], attempt_count: int) -> Dict[str, Any]:
        """
        Update the state based on the latest action and response.
        
        Args:
            action: Action taken
            payload: Payload used
            response: Response received
            analysis: Response analysis
            attempt_count: Current attempt count
            
        Returns:
            Updated state dictionary
        """
        # Extract payload features
        self.current_payload_features = self.payload_analyzer.extract_features(payload)
        
        # Extract response features
        self.response_features = self.response_analyzer.extract_features(response, analysis)
        
        # Update WAF features
        self._update_waf_features(response, analysis)
        
        # Update injection features
        self._update_injection_features(analysis, payload)
        
        # Update context features
        self._update_context_features(attempt_count, action)
        
        # Update history
        self._update_history(action, payload, response, analysis.get('injection_detected', False))
        
        # Update history features
        self._update_history_features()
        
        # Build state vector
        self._build_state_vector()
        
        return {
            'waf_detected': self.waf_features[0] > 0,
            'baseline_response': None,  # Maintained from initialization
            'attempt_count': attempt_count,
            'injection_detected': analysis.get('injection_detected', False),
            'current_payload': payload,
            'last_response': response,
            'waf_triggered': analysis.get('waf_triggered', False),
            'error_detected': analysis.get('error_detected', False)
        }
    
    def _update_waf_features(self, response: Dict[str, Any], analysis: Dict[str, Any]):
        """
        Update WAF-related features.
        
        Args:
            response: Response dictionary
            analysis: Analysis results
        """
        # WAF trigger indicator
        if analysis.get('waf_triggered', False):
            self.waf_features[1] = 1.0
        
        # Status code patterns indicating WAF
        status_code = response.get('status_code', 200)
        if status_code in [403, 406, 501, 503]:
            self.waf_features[2] = 1.0
        
        # Response time anomaly (could indicate WAF processing)
        response_time = response.get('response_time', 0)
        if response_time > 2.0:
            self.waf_features[3] = min(response_time / 10.0, 1.0)
        
        # Content-based WAF indicators
        content = response.get('content', '').lower()
        waf_keywords = ['blocked', 'forbidden', 'access denied', 'security', 'malicious']
        if any(keyword in content for keyword in waf_keywords):
            self.waf_features[4] = 1.0
    
    def _update_injection_features(self, analysis: Dict[str, Any], payload: str):
        """
        Update injection-related features.
        
        Args:
            analysis: Analysis results
            payload: Current payload
        """
        # Injection success indicators
        self.injection_features[0] = 1.0 if analysis.get('injection_detected', False) else 0.0
        self.injection_features[1] = 1.0 if analysis.get('error_detected', False) else 0.0
        self.injection_features[2] = 1.0 if analysis.get('blind_injection_possible', False) else 0.0
        
        # Payload type indicators
        payload_lower = payload.lower()
        self.injection_features[3] = 1.0 if 'union' in payload_lower else 0.0
        self.injection_features[4] = 1.0 if 'select' in payload_lower else 0.0
        self.injection_features[5] = 1.0 if any(keyword in payload_lower for keyword in ['sleep', 'waitfor', 'benchmark']) else 0.0
        self.injection_features[6] = 1.0 if any(keyword in payload_lower for keyword in ['and', 'or', '1=1', '1=2']) else 0.0
        
        # Encoding indicators
        self.injection_features[7] = 1.0 if '%' in payload else 0.0  # URL encoded
        self.injection_features[8] = 1.0 if '0x' in payload_lower else 0.0  # Hex encoded
        self.injection_features[9] = 1.0 if any(char in payload for char in ['<', '>', '"', "'"]) else 0.0  # Special chars
    
    def _update_context_features(self, attempt_count: int, action: int):
        """
        Update context-related features.
        
        Args:
            attempt_count: Current attempt count
            action: Current action
        """
        # Attempt progress
        self.context_features[0] = min(attempt_count / 100.0, 1.0)
        
        # Action type (normalized)
        self.context_features[1] = action / 20.0  # Assuming max 20 actions
        
        # Success rate so far
        successful_attempts = sum(1 for reward in self.reward_history if reward > 0)
        if attempt_count > 0:
            self.context_features[2] = successful_attempts / attempt_count
        
        # Recent failure streak
        if len(self.reward_history) >= 5:
            recent_failures = sum(1 for reward in list(self.reward_history)[-5:] if reward <= 0)
            self.context_features[3] = recent_failures / 5.0
    
    def _update_history(self, action: int, payload: str, response: Dict[str, Any], success: bool):
        """
        Update history tracking.
        
        Args:
            action: Action taken
            payload: Payload used
            response: Response received
            success: Whether injection was successful
        """
        self.action_history.append(action)
        self.payload_history.append(payload)
        self.response_history.append(response)
        self.reward_history.append(1.0 if success else 0.0)
    
    def _update_history_features(self):
        """
        Update features based on historical data.
        """
        if len(self.action_history) == 0:
            return
        
        # Action diversity
        unique_actions = len(set(self.action_history))
        self.history_features[0] = unique_actions / len(self.action_history)
        
        # Recent action distribution
        if len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]
            action_counts = {}
            for action in recent_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Most frequent recent action
            most_frequent_action = max(action_counts, key=action_counts.get)
            self.history_features[1] = most_frequent_action / 20.0
            
            # Action repetition
            self.history_features[2] = action_counts[most_frequent_action] / 5.0
        
        # Response time trend
        if len(self.response_history) >= 3:
            recent_times = [r.get('response_time', 0) for r in list(self.response_history)[-3:]]
            if len(recent_times) > 1:
                trend = recent_times[-1] - recent_times[0]
                self.history_features[3] = max(-1.0, min(1.0, trend / 5.0))  # Normalized trend
        
        # Success rate trend
        if len(self.reward_history) >= 10:
            first_half = list(self.reward_history)[:5]
            second_half = list(self.reward_history)[-5:]
            first_half_success = sum(first_half) / 5.0
            second_half_success = sum(second_half) / 5.0
            self.history_features[4] = second_half_success - first_half_success
    
    def _build_state_vector(self):
        """
        Build the complete state vector from all feature components.
        """
        self.state = np.concatenate([
            self.current_payload_features,  # 20 features
            self.response_features,         # 15 features
            self.waf_features,             # 10 features
            self.injection_features,       # 15 features
            self.context_features,         # 10 features
            self.history_features          # 30 features
        ])
        
        # Ensure state vector is the correct size
        if len(self.state) != self.state_size:
            # Pad or truncate to match expected size
            if len(self.state) < self.state_size:
                padding = np.zeros(self.state_size - len(self.state))
                self.state = np.concatenate([self.state, padding])
            else:
                self.state = self.state[:self.state_size]
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state vector.
        
        Returns:
            Current state as numpy array
        """
        return self.state.copy()
    
    def get_state_size(self) -> int:
        """
        Get the size of the state vector.
        
        Returns:
            State vector size
        """
        return self.state_size


class PayloadAnalyzer:
    """
    Analyzes payloads to extract numerical features.
    """
    
    def extract_features(self, payload: str) -> np.ndarray:
        """
        Extract features from a payload string.
        
        Args:
            payload: Payload to analyze
            
        Returns:
            Feature vector
        """
        features = np.zeros(20)
        
        if not payload:
            return features
        
        payload_lower = payload.lower()
        
        # Basic payload characteristics
        features[0] = len(payload) / 1000.0  # Normalized length
        features[1] = payload.count("'") / 10.0  # Single quotes
        features[2] = payload.count('"') / 10.0  # Double quotes
        features[3] = payload.count('(') / 10.0  # Parentheses
        features[4] = payload.count('%') / 20.0  # URL encoding
        
        # SQL keywords
        sql_keywords = ['select', 'union', 'insert', 'update', 'delete', 'drop', 'create']
        for i, keyword in enumerate(sql_keywords):
            if i < 7:  # Limit to available feature slots
                features[5 + i] = 1.0 if keyword in payload_lower else 0.0
        
        # Special patterns
        features[12] = 1.0 if '1=1' in payload else 0.0
        features[13] = 1.0 if '1=2' in payload else 0.0
        features[14] = 1.0 if 'sleep(' in payload_lower else 0.0
        features[15] = 1.0 if 'waitfor' in payload_lower else 0.0
        features[16] = 1.0 if 'benchmark(' in payload_lower else 0.0
        
        # Encoding patterns
        features[17] = 1.0 if '0x' in payload_lower else 0.0
        features[18] = 1.0 if '\\x' in payload else 0.0
        features[19] = 1.0 if '--' in payload or '/*' in payload else 0.0
        
        return features


class ResponseAnalyzer:
    """
    Analyzes HTTP responses to extract numerical features.
    """
    
    def extract_features(self, response: Dict[str, Any], analysis: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a response and its analysis.
        
        Args:
            response: Response dictionary
            analysis: Analysis results
            
        Returns:
            Feature vector
        """
        features = np.zeros(15)
        
        # Basic response characteristics
        features[0] = min(response.get('status_code', 200) / 500.0, 1.0)
        features[1] = min(len(response.get('content', '')) / 10000.0, 1.0)
        features[2] = min(response.get('response_time', 0) / 10.0, 1.0)
        
        # Analysis results
        features[3] = 1.0 if analysis.get('injection_detected', False) else 0.0
        features[4] = 1.0 if analysis.get('waf_triggered', False) else 0.0
        features[5] = 1.0 if analysis.get('error_detected', False) else 0.0
        features[6] = 1.0 if analysis.get('blind_injection_possible', False) else 0.0
        features[7] = 1.0 if analysis.get('response_anomaly', False) else 0.0
        
        # Content analysis
        content = response.get('content', '').lower()
        features[8] = 1.0 if 'error' in content else 0.0
        features[9] = 1.0 if 'mysql' in content else 0.0
        features[10] = 1.0 if 'oracle' in content else 0.0
        features[11] = 1.0 if 'postgresql' in content else 0.0
        features[12] = 1.0 if 'sql server' in content else 0.0
        
        # Headers analysis
        headers = response.get('headers', {})
        features[13] = 1.0 if 'server' in headers else 0.0
        features[14] = 1.0 if any('security' in str(v).lower() for v in headers.values()) else 0.0
        
        return features
