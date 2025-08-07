"""
Action space definition for SQL injection RL agent
"""

from enum import Enum
from typing import Dict, List, Tuple, Any
import json


class ActionType(Enum):
    """
    Types of actions the RL agent can take.
    """
    # Payload manipulation actions
    MODIFY_PAYLOAD = "modify_payload"
    ENCODE_PAYLOAD = "encode_payload"
    OBFUSCATE_PAYLOAD = "obfuscate_payload"
    
    # WAF bypass techniques
    CASE_VARIATION = "case_variation"
    COMMENT_INSERTION = "comment_insertion"
    WHITESPACE_MANIPULATION = "whitespace_manipulation"
    KEYWORD_REPLACEMENT = "keyword_replacement"
    
    # Testing strategies
    BLIND_INJECTION = "blind_injection"
    UNION_INJECTION = "union_injection"
    ERROR_BASED = "error_based"
    TIME_BASED = "time_based"
    
    # Advanced techniques
    DOUBLE_ENCODING = "double_encoding"
    HTTP_PARAMETER_POLLUTION = "hpp"
    CHUNKED_ENCODING = "chunked_encoding"
    
    # Termination
    TERMINATE = "terminate"


class ActionSpace:
    """
    Defines the action space for the SQL injection RL agent.
    """
    
    def __init__(self):
        """
        Initialize the action space with predefined actions.
        """
        self.actions = self._define_actions()
        self.action_count = len(self.actions)
        self.action_to_index = {action['name']: i for i, action in enumerate(self.actions)}
        self.index_to_action = {i: action for i, action in enumerate(self.actions)}
    
    def _define_actions(self) -> List[Dict[str, Any]]:
        """
        Define all possible actions for the agent.
        
        Returns:
            List of action dictionaries
        """
        actions = [
            # Basic payload modifications
            {
                'name': 'append_comment',
                'type': ActionType.MODIFY_PAYLOAD,
                'description': 'Append SQL comment to payload',
                'parameters': {'comment_type': ['--', '/*', '#']}
            },
            {
                'name': 'prepend_comment',
                'type': ActionType.MODIFY_PAYLOAD,
                'description': 'Prepend SQL comment to payload',
                'parameters': {'comment_type': ['--', '/*', '#']}
            },
            {
                'name': 'insert_comment',
                'type': ActionType.COMMENT_INSERTION,
                'description': 'Insert comment within payload',
                'parameters': {'position': 'dynamic', 'comment_type': ['--', '/*', '#']}
            },
            
            # Encoding actions
            {
                'name': 'url_encode',
                'type': ActionType.ENCODE_PAYLOAD,
                'description': 'Apply URL encoding to payload',
                'parameters': {'encode_type': ['single', 'double']}
            },
            {
                'name': 'hex_encode',
                'type': ActionType.ENCODE_PAYLOAD,
                'description': 'Apply hexadecimal encoding',
                'parameters': {'prefix': ['0x', '\\x']}
            },
            {
                'name': 'unicode_encode',
                'type': ActionType.ENCODE_PAYLOAD,
                'description': 'Apply Unicode encoding',
                'parameters': {'format': ['%u', '\\u']}
            },
            
            # Case manipulation
            {
                'name': 'randomize_case',
                'type': ActionType.CASE_VARIATION,
                'description': 'Randomize character case',
                'parameters': {'probability': 0.5}
            },
            {
                'name': 'alternate_case',
                'type': ActionType.CASE_VARIATION,
                'description': 'Alternate between upper/lower case',
                'parameters': {}
            },
            
            # Whitespace manipulation
            {
                'name': 'add_whitespace',
                'type': ActionType.WHITESPACE_MANIPULATION,
                'description': 'Add random whitespace',
                'parameters': {'chars': [' ', '\\t', '\\n', '\\r']}
            },
            {
                'name': 'replace_spaces',
                'type': ActionType.WHITESPACE_MANIPULATION,
                'description': 'Replace spaces with alternatives',
                'parameters': {'replacements': ['/**/', '\\t', '+', '%20']}
            },
            
            # Keyword obfuscation
            {
                'name': 'keyword_concat',
                'type': ActionType.KEYWORD_REPLACEMENT,
                'description': 'Use string concatenation for keywords',
                'parameters': {'method': ['concat', 'chr', 'char']}
            },
            {
                'name': 'keyword_equivalent',
                'type': ActionType.KEYWORD_REPLACEMENT,
                'description': 'Replace keywords with equivalents',
                'parameters': {
                    'replacements': {
                        'SELECT': ['DISTINCT', 'ALL'],
                        'UNION': ['UNION ALL', 'UNION DISTINCT'],
                        'AND': ['&&'],
                        'OR': ['||']
                    }
                }
            },
            
            # Advanced obfuscation
            {
                'name': 'function_obfuscation',
                'type': ActionType.OBFUSCATE_PAYLOAD,
                'description': 'Obfuscate using SQL functions',
                'parameters': {'functions': ['CHAR', 'CHR', 'ASCII', 'CONCAT']}
            },
            {
                'name': 'arithmetic_obfuscation',
                'type': ActionType.OBFUSCATE_PAYLOAD,
                'description': 'Use arithmetic expressions',
                'parameters': {'operations': ['+', '-', '*', '/', '%']}
            },
            
            # Injection techniques
            {
                'name': 'union_select',
                'type': ActionType.UNION_INJECTION,
                'description': 'Attempt UNION SELECT injection',
                'parameters': {'columns': 'dynamic', 'null_padding': True}
            },
            {
                'name': 'error_based_injection',
                'type': ActionType.ERROR_BASED,
                'description': 'Trigger error-based information disclosure',
                'parameters': {'error_functions': ['EXTRACTVALUE', 'UPDATEXML', 'EXP']}
            },
            {
                'name': 'time_delay',
                'type': ActionType.TIME_BASED,
                'description': 'Inject time delay functions',
                'parameters': {'delay_functions': ['SLEEP', 'WAITFOR', 'BENCHMARK']}
            },
            {
                'name': 'boolean_blind',
                'type': ActionType.BLIND_INJECTION,
                'description': 'Boolean-based blind injection',
                'parameters': {'conditions': ['TRUE', 'FALSE', '1=1', '1=2']}
            },
            
            # WAF-specific bypass techniques
            {
                'name': 'double_url_encode',
                'type': ActionType.DOUBLE_ENCODING,
                'description': 'Apply double URL encoding',
                'parameters': {}
            },
            {
                'name': 'http_param_pollution',
                'type': ActionType.HTTP_PARAMETER_POLLUTION,
                'description': 'Use HTTP parameter pollution',
                'parameters': {'duplicate_params': True}
            },
            {
                'name': 'chunked_transfer',
                'type': ActionType.CHUNKED_ENCODING,
                'description': 'Use chunked transfer encoding',
                'parameters': {'chunk_size': 'random'}
            },
            
            # Buffer overflow attempts
            {
                'name': 'buffer_overflow',
                'type': ActionType.MODIFY_PAYLOAD,
                'description': 'Attempt buffer overflow',
                'parameters': {'padding_char': 'A', 'length': 'dynamic'}
            },
            
            # Termination action
            {
                'name': 'terminate',
                'type': ActionType.TERMINATE,
                'description': 'Terminate the attack sequence',
                'parameters': {}
            }
        ]
        
        return actions
    
    def get_action_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get action by its index.
        
        Args:
            index: Action index
            
        Returns:
            Action dictionary
        """
        return self.index_to_action.get(index, None)
    
    def get_action_index(self, action_name: str) -> int:
        """
        Get action index by name.
        
        Args:
            action_name: Name of the action
            
        Returns:
            Action index or -1 if not found
        """
        return self.action_to_index.get(action_name, -1)
    
    def get_actions_by_type(self, action_type: ActionType) -> List[Dict[str, Any]]:
        """
        Get all actions of a specific type.
        
        Args:
            action_type: Type of actions to retrieve
            
        Returns:
            List of actions of the specified type
        """
        return [action for action in self.actions if action['type'] == action_type]
    
    def get_compatible_actions(self, current_state: Dict[str, Any]) -> List[int]:
        """
        Get actions that are compatible with the current state.
        
        Args:
            current_state: Current environment state
            
        Returns:
            List of compatible action indices
        """
        compatible_actions = []
        
        for i, action in enumerate(self.actions):
            if self._is_action_compatible(action, current_state):
                compatible_actions.append(i)
        
        return compatible_actions
    
    def _is_action_compatible(self, action: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Check if an action is compatible with the current state.
        
        Args:
            action: Action to check
            state: Current state
            
        Returns:
            True if action is compatible, False otherwise
        """
        # Basic compatibility checks
        
        # Don't allow termination if no attempts have been made
        if action['type'] == ActionType.TERMINATE and state.get('attempt_count', 0) == 0:
            return False
        
        # Don't allow encoding if already heavily encoded
        if action['type'] == ActionType.ENCODE_PAYLOAD and state.get('encoding_count', 0) > 3:
            return False
        
        # Check if WAF is detected before applying bypass techniques
        waf_bypass_types = [
            ActionType.DOUBLE_ENCODING,
            ActionType.HTTP_PARAMETER_POLLUTION,
            ActionType.CHUNKED_ENCODING
        ]
        if action['type'] in waf_bypass_types and not state.get('waf_detected', False):
            return False
        
        return True
    
    def get_action_description(self, index: int) -> str:
        """
        Get human-readable description of an action.
        
        Args:
            index: Action index
            
        Returns:
            Action description
        """
        action = self.get_action_by_index(index)
        return action['description'] if action else "Unknown action"
    
    def export_action_space(self, filepath: str):
        """
        Export action space definition to JSON file.
        
        Args:
            filepath: Path to save the action space
        """
        # Convert ActionType enums to strings for JSON serialization
        actions_serializable = []
        for action in self.actions:
            action_copy = action.copy()
            action_copy['type'] = action['type'].value
            actions_serializable.append(action_copy)
        
        with open(filepath, 'w') as f:
            json.dump(actions_serializable, f, indent=2)
    
    def get_action_count(self) -> int:
        """
        Get total number of actions in the action space.
        
        Returns:
            Number of actions
        """
        return self.action_count
