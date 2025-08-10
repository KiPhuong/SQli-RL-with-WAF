"""
Simple State Manager for SQL Injection RL Agent
Provides 50-feature state representation focused on essential context
"""

import numpy as np
from typing import Dict, List, Any, Optional


class SimpleStateManager:
    """Manages simple but effective state representation for SQL injection RL"""
    
    def __init__(self):
        self.state_size = 50
        self.last_response = None
        self.recent_blocks = 0
        self.total_reward = 0.0
        self.has_success = False
        self.step_count = 0
        
    def build_state(self, current_payload: str, response: Optional[Dict] = None, 
                   is_blocked: bool = False, bypass_applied: bool = False, 
                   bypass_method: Optional[str] = None, step_count: int = 0, 
                   max_steps: int = 50, reward: float = 0.0) -> np.ndarray:
        """
        Build state vector from current context
        
        Returns:
            np.ndarray: State vector of 50 features
        """
        # Update internal tracking
        self.step_count = step_count
        if response:
            self.last_response = response
        
        if is_blocked:
            self.recent_blocks = min(self.recent_blocks + 1, 5)
        else:
            self.recent_blocks = max(self.recent_blocks - 1, 0)
        
        self.total_reward += reward
        if reward > 0.8:  # High reward indicates success
            self.has_success = True
        
        # Extract feature groups
        payload_features = self._extract_payload_features(current_payload)      # 20
        response_features = self._extract_response_features(response)           # 15
        waf_features = self._extract_waf_features(is_blocked, bypass_applied, 
                                                 bypass_method)                 # 10
        progress_features = self._extract_progress_features(step_count, max_steps) # 5
        
        # Combine all features (total: 50)
        state = np.concatenate([
            payload_features,
            response_features, 
            waf_features,
            progress_features
        ])
        
        return state.astype(np.float32)
    
    def _extract_payload_features(self, payload: str) -> np.ndarray:
        """Extract features from current payload (20 features)"""
        features = []
        
        # Basic payload characteristics
        features.append(len(payload) / 100.0)                   # [0] Length
        features.append(payload.count("'") / 5.0)               # [1] Single quotes
        features.append(payload.count(' ') / 10.0)              # [2] Spaces
        features.append(payload.count('(') / 5.0)               # [3] Parentheses
        
        # Key SQL keywords (binary flags)
        keywords = ['SELECT', 'UNION', 'FROM', 'WHERE', 'AND', 'OR', 
                   'INSERT', 'UPDATE', 'DELETE', 'DROP']
        for keyword in keywords:  # [4-13]
            features.append(1.0 if keyword in payload.upper() else 0.0)
        
        # Injection indicators
        features.append(1.0 if 'UNION' in payload.upper() else 0.0)     # [14] Union-based
        features.append(1.0 if 'SLEEP' in payload.upper() else 0.0)     # [15] Time-based
        features.append(1.0 if ' OR ' in payload.upper() else 0.0)      # [16] Boolean-based
        features.append(1.0 if '--' in payload else 0.0)               # [17] Comments
        features.append(1.0 if ';' in payload else 0.0)                # [18] Statement separator
        features.append(payload.count('%') / 10.0)                     # [19] URL encoding
        
        return np.array(features, dtype=np.float32)
    
    def _extract_response_features(self, response: Optional[Dict]) -> np.ndarray:
        """Extract features from server response (15 features)"""
        features = []
        
        if not response:
            return np.zeros(15, dtype=np.float32)
        
        # HTTP basics
        features.append(response.get('status_code', 0) / 1000.0)        # [0] Status code
        features.append(min(response.get('content_length', 0) / 5000.0, 1.0))  # [1] Content length
        features.append(min(response.get('response_time', 0) / 5.0, 1.0))      # [2] Response time
        
        # Error detection
        content = response.get('content', '').lower()
        features.append(1.0 if 'error' in content else 0.0)     # [3] Has error
        features.append(1.0 if 'mysql' in content else 0.0)     # [4] MySQL error
        features.append(1.0 if 'column' in content else 0.0)    # [5] Column error
        features.append(1.0 if 'table' in content else 0.0)     # [6] Table error
        features.append(1.0 if 'syntax' in content else 0.0)    # [7] Syntax error
        
        # Success indicators
        features.append(1.0 if 'fetch_array' in content else 0.0)       # [8] MySQL success
        features.append(1.0 if response.get('response_time', 0) > 3.0 else 0.0) # [9] Time delay
        
        # Response patterns
        status_code = response.get('status_code', 0)
        features.append(1.0 if status_code == 200 else 0.0)     # [10] Success status
        features.append(1.0 if status_code == 403 else 0.0)     # [11] Forbidden
        features.append(1.0 if status_code == 500 else 0.0)     # [12] Server error
        features.append(1.0 if 'blocked' in content else 0.0)   # [13] Blocked message
        features.append(1.0 if 'forbidden' in content else 0.0) # [14] Forbidden message
        
        return np.array(features, dtype=np.float32)
    
    def _extract_waf_features(self, is_blocked: bool, bypass_applied: bool, 
                             bypass_method: Optional[str]) -> np.ndarray:
        """Extract WAF-related features (10 features)"""
        features = []
        
        # Current WAF status
        features.append(1.0 if is_blocked else 0.0)             # [0] Currently blocked
        features.append(1.0 if bypass_applied else 0.0)         # [1] Bypass applied
        features.append(self.recent_blocks / 5.0)               # [2] Recent block rate
        
        # Bypass method (one-hot encoding)
        bypass_methods = ['case_variation', 'comment_insertion', 'url_encoding', 
                         'hex_encoding', 'unicode_encoding', 'none']
        current_method = bypass_method if bypass_method else 'none'
        
        for method in bypass_methods:  # [3-8]
            features.append(1.0 if current_method == method else 0.0)
        
        # WAF sensitivity
        features.append(min(self.recent_blocks / 10.0, 1.0))    # [9] WAF aggressiveness
        
        return np.array(features, dtype=np.float32)
    
    def _extract_progress_features(self, step_count: int, max_steps: int) -> np.ndarray:
        """Extract episode progress features (5 features)"""
        features = []
        
        features.append(step_count / max_steps)                  # [0] Episode progress
        features.append(self.total_reward / 10.0)               # [1] Cumulative reward
        features.append(1.0 if self.has_success else 0.0)       # [2] Found success
        features.append(1.0 if step_count < max_steps * 0.5 else 0.0)  # [3] Early phase
        features.append(1.0 if self.total_reward > 0 else 0.0)  # [4] Positive progress
        
        return np.array(features, dtype=np.float32)
    
    def reset(self):
        """Reset state manager for new episode"""
        self.last_response = None
        self.recent_blocks = 0
        self.total_reward = 0.0
        self.has_success = False
        self.step_count = 0
    
    def get_state_size(self) -> int:
        """Get state vector size"""
        return self.state_size
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for debugging"""
        names = []
        
        # Payload features (20)
        names.extend([
            'payload_length', 'single_quotes', 'spaces', 'parentheses',
            'has_select', 'has_union', 'has_from', 'has_where', 'has_and', 'has_or',
            'has_insert', 'has_update', 'has_delete', 'has_drop',
            'union_based', 'time_based', 'boolean_based', 'has_comments', 'has_semicolon', 'url_encoding'
        ])
        
        # Response features (15)
        names.extend([
            'status_code', 'content_length', 'response_time',
            'has_error', 'mysql_error', 'column_error', 'table_error', 'syntax_error',
            'mysql_success', 'time_delay',
            'status_200', 'status_403', 'status_500', 'blocked_msg', 'forbidden_msg'
        ])
        
        # WAF features (10)
        names.extend([
            'is_blocked', 'bypass_applied', 'recent_blocks',
            'bypass_case', 'bypass_comment', 'bypass_url', 'bypass_hex', 'bypass_unicode', 'bypass_none',
            'waf_aggressiveness'
        ])
        
        # Progress features (5)
        names.extend([
            'episode_progress', 'cumulative_reward', 'found_success', 'early_phase', 'positive_progress'
        ])
        
        return names
    
    def debug_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Debug state vector by showing feature values"""
        feature_names = self.get_feature_names()
        
        debug_info = {
            'state_size': len(state),
            'payload_features': dict(zip(feature_names[:20], state[:20])),
            'response_features': dict(zip(feature_names[20:35], state[20:35])),
            'waf_features': dict(zip(feature_names[35:45], state[35:45])),
            'progress_features': dict(zip(feature_names[45:50], state[45:50]))
        }
        
        return debug_info


def test_simple_state():
    """Test the SimpleStateManager"""
    print("🧪 Testing SimpleStateManager")
    print("=" * 50)
    
    state_manager = SimpleStateManager()
    
    # Test initial state
    initial_state = state_manager.build_state("", step_count=0, max_steps=50)
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Initial state range: [{initial_state.min():.3f}, {initial_state.max():.3f}]")
    
    # Test with payload
    test_payload = "SELECT * FROM users WHERE id=1 UNION SELECT 1,2,3--"
    test_response = {
        'status_code': 200,
        'content': 'Unknown column \'1\' in \'where clause\'',
        'content_length': 1234,
        'response_time': 0.5
    }
    
    state = state_manager.build_state(
        current_payload=test_payload,
        response=test_response,
        is_blocked=False,
        bypass_applied=True,
        bypass_method='comment_insertion',
        step_count=5,
        max_steps=50,
        reward=0.7
    )
    
    print(f"\nTest state shape: {state.shape}")
    print(f"Test state range: [{state.min():.3f}, {state.max():.3f}]")
    
    # Debug state
    debug_info = state_manager.debug_state(state)
    print(f"\nPayload features: {debug_info['payload_features']}")
    print(f"Response features: {debug_info['response_features']}")
    print(f"WAF features: {debug_info['waf_features']}")
    print(f"Progress features: {debug_info['progress_features']}")
    
    print("\n✅ SimpleStateManager test completed!")


if __name__ == "__main__":
    test_simple_state()
