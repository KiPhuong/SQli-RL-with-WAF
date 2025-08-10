"""
Token Generator Module for SQL Injection RL Agent
Handles token generation and payload construction with WAF bypass integration
"""

import random
from typing import List, Dict, Any, Optional
from .waf.bypass_methods import BypassMethods
from .waf.waf_detector import WAFDetector


class TokenGenerator:
    """
    Generates SQL injection tokens and constructs payloads with WAF bypass capabilities.
    """
    
    def __init__(self):
        """Initialize the token generator."""
        self.bypass_methods = BypassMethods()
        self.waf_detector = WAFDetector()
        
        # SQL injection token vocabulary
        self.tokens = {
            # Basic SQL keywords
            'SELECT': 0, 'FROM': 1, 'WHERE': 2, 'AND': 3, 'OR': 4, 'UNION': 5,
            'INSERT': 6, 'UPDATE': 7, 'DELETE': 8, 'DROP': 9, 'CREATE': 10,
            
            # SQL operators and symbols
            '=': 11, '!=': 12, '<': 13, '>': 14, '<=': 15, '>=': 16,
            '+': 17, '-': 18, '*': 19, '/': 20, '%': 21,
            
            # SQL injection specific tokens
            "'": 22, '"': 23, '--': 24, '/*': 25, '*/': 26, '#': 27,
            ';': 28, '(': 29, ')': 30, ',': 31, '.': 32,
            
            # Common SQL injection payloads components
            '1': 33, '0': 34, 'NULL': 35, 'TRUE': 36, 'FALSE': 37,
            'SLEEP': 38, 'BENCHMARK': 39, 'WAITFOR': 40, 'DELAY': 41,
            
            # Database specific functions
            'VERSION': 42, 'USER': 43, 'DATABASE': 44, 'SCHEMA': 45,
            'TABLE_NAME': 46, 'COLUMN_NAME': 47, 'CONCAT': 48, 'SUBSTRING': 49,
            
            # Error-based injection tokens
            'EXTRACTVALUE': 50, 'UPDATEXML': 51, 'XMLTYPE': 52, 'CAST': 53,
            'CONVERT': 54, 'CHAR': 55, 'ASCII': 56, 'HEX': 57,
            
            # Union-based injection tokens
            'ALL': 58, 'DISTINCT': 59, 'ORDER': 60, 'BY': 61, 'GROUP': 62,
            'HAVING': 63, 'LIMIT': 64, 'OFFSET': 65,
            
            # Boolean-based injection tokens
            'LIKE': 66, 'REGEXP': 67, 'RLIKE': 68, 'SOUNDS': 69, 'MATCH': 70,
            'AGAINST': 71, 'BINARY': 72, 'COLLATE': 73,
            
            # Time-based injection tokens
            'IF': 74, 'CASE': 75, 'WHEN': 76, 'THEN': 77, 'ELSE': 78, 'END': 79,
            
            # Special characters and encodings
            'SPACE': 80, 'TAB': 81, 'NEWLINE': 82, 'RETURN': 83,
            'PERCENT': 84, 'UNDERSCORE': 85, 'BACKSLASH': 86,
            
            # Padding and termination tokens
            'PAD': 87, 'END_TOKEN': 88
        }
        
        # Reverse mapping for token to string conversion
        self.token_to_string = {v: k for k, v in self.tokens.items()}
        
        # Token categories for intelligent generation
        self.token_categories = {
            'keywords': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'operators': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'symbols': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            'values': [33, 34, 35, 36, 37],
            'functions': [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            'injection_specific': [50, 51, 52, 53, 54, 55, 56, 57],
            'union_tokens': [58, 59, 60, 61, 62, 63, 64, 65],
            'boolean_tokens': [66, 67, 68, 69, 70, 71, 72, 73],
            'time_tokens': [74, 75, 76, 77, 78, 79],
            'special_chars': [80, 81, 82, 83, 84, 85, 86],
            'control': [87, 88]
        }
        
        self.vocab_size = len(self.tokens)
        self.max_payload_length = 200
        
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def token_to_text(self, token_id: int) -> str:
        """Convert token ID to text representation."""
        if token_id in self.token_to_string:
            token_str = self.token_to_string[token_id]
            
            # Handle special tokens
            if token_str == 'SPACE':
                return ' '
            elif token_str == 'TAB':
                return '\t'
            elif token_str == 'NEWLINE':
                return '\n'
            elif token_str == 'RETURN':
                return '\r'
            elif token_str == 'PERCENT':
                return '%'
            elif token_str == 'UNDERSCORE':
                return '_'
            elif token_str == 'BACKSLASH':
                return '\\'
            elif token_str == 'PAD':
                return ''
            elif token_str == 'END_TOKEN':
                return ''
            else:
                return token_str
        else:
            return ''
    
    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to list of token IDs."""
        tokens = []
        words = text.split()
        
        for word in words:
            if word.upper() in self.tokens:
                tokens.append(self.tokens[word.upper()])
            else:
                # Try to match individual characters
                for char in word:
                    if char in self.tokens:
                        tokens.append(self.tokens[char])
                    elif char.upper() in self.tokens:
                        tokens.append(self.tokens[char.upper()])
        
        return tokens
    
    def generate_payload_from_tokens(self, token_sequence: List[int], 
                                   waf_detected: bool = False,
                                   bypass_enabled: bool = True) -> str:
        """
        Generate SQL injection payload from token sequence.
        
        Args:
            token_sequence: List of token IDs
            waf_detected: Whether WAF is detected
            bypass_enabled: Whether to apply bypass techniques
            
        Returns:
            Generated payload string
        """
        payload_parts = []
        
        for token_id in token_sequence:
            if token_id == self.tokens['END_TOKEN']:
                break
            elif token_id == self.tokens['PAD']:
                continue
            
            token_text = self.token_to_text(token_id)
            
            # Apply WAF bypass if needed
            if waf_detected and bypass_enabled and token_text:
                token_text = self._apply_bypass_to_token(token_text)
            
            if token_text:
                payload_parts.append(token_text)
        
        return ' '.join(payload_parts)
    
    def _apply_bypass_to_token(self, token: str) -> str:
        """
        Apply bypass techniques to a single token.
        
        Args:
            token: Original token
            
        Returns:
            Modified token with bypass applied
        """
        # Randomly select bypass method
        bypass_methods = ['case_variation', 'comment_insertion', 'url_encode', 'hex_encode']
        selected_method = random.choice(bypass_methods)
        
        try:
            if selected_method == 'case_variation':
                return self.bypass_methods.apply_bypass_method(token, 'case_variation', variation_method='random')
            elif selected_method == 'comment_insertion':
                return self.bypass_methods.apply_bypass_method(token, 'comment_insertion')
            elif selected_method == 'url_encode':
                return self.bypass_methods.apply_bypass_method(token, 'url_encode')
            elif selected_method == 'hex_encode':
                return self.bypass_methods.apply_bypass_method(token, 'hex_encode')
            else:
                return token
        except Exception:
            return token
    
    def add_token_to_state(self, current_state: List[int], new_token: int) -> List[int]:
        """
        Add a new token to the current state.
        
        Args:
            current_state: Current token sequence
            new_token: New token to add
            
        Returns:
            Updated token sequence
        """
        # Remove padding tokens from the end
        while current_state and current_state[-1] == self.tokens['PAD']:
            current_state.pop()
        
        # Add new token
        current_state.append(new_token)
        
        # Ensure state doesn't exceed max length
        if len(current_state) > self.max_payload_length:
            current_state = current_state[-self.max_payload_length:]
        
        # Pad to fixed length
        while len(current_state) < self.max_payload_length:
            current_state.append(self.tokens['PAD'])
        
        return current_state
    
    def normalize_state(self, token_sequence: List[int]) -> List[int]:
        """
        Normalize token sequence to fixed length (200 tokens).
        
        Args:
            token_sequence: Input token sequence
            
        Returns:
            Normalized token sequence of length 200
        """
        normalized = token_sequence.copy()
        
        # Truncate if too long
        if len(normalized) > self.max_payload_length:
            normalized = normalized[:self.max_payload_length]
        
        # Pad if too short
        while len(normalized) < self.max_payload_length:
            normalized.append(self.tokens['PAD'])
        
        return normalized
    
    def get_initial_state(self) -> List[int]:
        """Get initial empty state."""
        return [self.tokens['PAD']] * self.max_payload_length
    
    def is_payload_complete(self, token_sequence: List[int]) -> bool:
        """Check if payload is complete (contains END_TOKEN)."""
        return self.tokens['END_TOKEN'] in token_sequence
    
    def get_random_token(self, category: Optional[str] = None) -> int:
        """
        Get a random token, optionally from a specific category.
        
        Args:
            category: Token category to sample from
            
        Returns:
            Random token ID
        """
        if category and category in self.token_categories:
            return random.choice(self.token_categories[category])
        else:
            return random.randint(0, self.vocab_size - 1)
