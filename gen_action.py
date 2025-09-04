"""
Gen-Action Module for SQL Injection RL Agent
Handles payload generation by appending tokens to current state
"""

import numpy as np
from typing import List, Dict, Any, Optional


class ActionSpace:
    """Manages the action space (token vocabulary) loaded from keywords.txt"""
    
    def __init__(self, keywords_file: str = "keywords.txt"):
        self.keywords_file = keywords_file
        self.tokens = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.load_keywords()
    
    def load_keywords(self):
        """Load keywords from file, remove duplicates, normalize, and create mappings"""
        try:
            with open(self.keywords_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            keywords = []
            seen = set()

            for line in lines:
                token = line.strip()

                # Bỏ comment và dòng rỗng
                if not token:
                    continue

                # Giữ nguyên ký tự đặc biệt, chuẩn hóa keyword về UPPERCASE
                if token.isalpha():  
                    token = token.upper()

                # Loại bỏ trùng lặp
                if token not in seen:
                    keywords.append(token)
                    seen.add(token)

            # Tạo mapping
            self.tokens = keywords
            self.token_to_id = {token: i for i, token in enumerate(keywords)}
            self.id_to_token = {i: token for i, token in enumerate(keywords)}

        except Exception as e:
            print(f"[ERROR] Failed to load keywords: {e}")

    
    def _create_default_tokens(self):
        """Create default token set if file not found"""
        default_tokens = [
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'UNION', 'INSERT', 'UPDATE', 'DELETE',
            '=', '!=', '<', '>', '<=', '>=', 'LIKE', 'IN', 'NOT', 'EXISTS',
            "'", '"', '--', '/*', '*/', '#', ';', '(', ')', ',', '.',
            '1', '0', 'TRUE', 'FALSE', 'NULL', 'SLEEP', 'WAITFOR', 'DELAY',
            'EXTRACTVALUE', 'UPDATEXML', 'CAST', 'CONVERT', 'CHAR', 'ASCII',
            'SPACE', 'TAB', 'PAD', 'END_TOKEN'
        ]
        
        self.tokens = default_tokens
        self.token_to_id = {token: i for i, token in enumerate(default_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(default_tokens)}
    
    def get_token_by_id(self, token_id: int) -> str:
        """Get token string by ID"""
        return self.id_to_token.get(token_id, 'UNK')
    
    def get_id_by_token(self, token: str) -> int:
        """Get token ID by string"""
        return self.token_to_id.get(token, -1)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokens)
    
    def get_all_tokens(self) -> List[str]:
        """Get all tokens"""
        return self.tokens.copy()


class PayloadGenerator:
    """Generates SQL injection payloads from token sequences"""
    
    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space
        self.max_payload_length = 200
    
    def token_to_text(self, token_id: int) -> str:
        """Convert token ID to actual text representation"""
        token = self.action_space.get_token_by_id(token_id)
        
        # Handle special tokens
        if token == 'SPACE':
            return ' '
        elif token == 'TAB':
            return '\t'
        elif token == 'NEWLINE':
            return '\n'
        elif token == 'RETURN':
            return '\r'
        elif token == 'PERCENT':
            return '%'
        elif token == 'UNDERSCORE':
            return '_'
        elif token == 'BACKSLASH':
            return '\\'
        elif token == 'PAD':
            return ''
        elif token == 'END_TOKEN':
            return ''
        else:
            return token
    
    def tokens_to_payload(self, token_sequence: List[int]) -> str:
        """Convert token sequence to payload string"""
        payload_parts = []
        
        for token_id in token_sequence:
            if token_id == self.action_space.get_id_by_token('END_TOKEN'):
                break
            elif token_id == self.action_space.get_id_by_token('PAD'):
                continue
            
            token_text = self.token_to_text(token_id)
            if token_text:
                payload_parts.append(token_text)
        
        return ' '.join(payload_parts)
    
    def payload_to_tokens(self, payload: str) -> List[int]:
        """Convert payload string to token sequence (for initialization)"""
        tokens = []
        words = payload.split()
        
        for word in words:
            token_id = self.action_space.get_id_by_token(word.upper())
            if token_id != -1:
                tokens.append(token_id)
            else:
                # Try to match individual characters
                for char in word:
                    char_id = self.action_space.get_id_by_token(char)
                    if char_id != -1:
                        tokens.append(char_id)
        
        return tokens


class GenAction:
    """Main Gen-Action module for payload generation"""

    def __init__(self, keywords_file: str = "keywords.txt"):
        self.action_space = ActionSpace(keywords_file)
        self.payload_generator = PayloadGenerator(self.action_space)
        self.state_length = 200  # Fixed state length for simplicity

        print(f"📊 Vocabulary size: {self.action_space.get_vocab_size()}")
        print(f"📏 State length: {self.state_length}")
    
    def create_initial_state(self) -> np.ndarray:
        """Create initial empty state (all PAD tokens)"""
        pad_token_id = self.action_space.get_id_by_token('PAD')
        if pad_token_id == -1:
            pad_token_id = 0  # Fallback

        return np.full(self.state_length, pad_token_id, dtype=np.float32)




    
    def state_to_payload(self, state: np.ndarray) -> str:
        """Convert state to payload string"""
        token_sequence = [int(token_id) for token_id in state]
        return self.payload_generator.tokens_to_payload(token_sequence)
    
    def append_token_to_state(self, current_state: np.ndarray, new_token_id: int) -> np.ndarray:
        """Append new token to current state"""
        new_state = current_state.copy()
        pad_token_id = self.action_space.get_id_by_token('PAD')
        
        # Find first PAD token and replace it
        for i, token_id in enumerate(new_state):
            if token_id == pad_token_id:
                new_state[i] = new_token_id
                break
        else:
            # If no PAD token found, shift left and add at end
            new_state = np.roll(new_state, -1)
            new_state[-1] = new_token_id
        
        return new_state
    
    def is_state_complete(self, state: np.ndarray) -> bool:
        """Check if state contains END_TOKEN"""
        end_token_id = self.action_space.get_id_by_token('END_TOKEN')
        return end_token_id in state
    
    def get_state_info(self, state: np.ndarray) -> Dict[str, Any]:
        """Get information about current state"""
        pad_token_id = self.action_space.get_id_by_token('PAD')
        end_token_id = self.action_space.get_id_by_token('END_TOKEN')
        
        # Count non-PAD tokens
        non_pad_tokens = np.sum(state != pad_token_id)
        
        # Check if complete
        is_complete = end_token_id in state
        
        # Get current payload
        payload = self.state_to_payload(state)
        
        return {
            'payload': payload,
            'length': non_pad_tokens,
            'is_complete': is_complete,
            'is_full': non_pad_tokens >= self.state_length,
            'payload_chars': len(payload)
        }
    
    def normalize_payload_length(self, payload: str) -> str:
        """Normalize payload to fixed length (pad or truncate)"""
        if len(payload) > self.state_length:
            return payload[:self.state_length]
        else:
            return payload.ljust(self.state_length)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.action_space.get_vocab_size()
    
    def get_action_space(self) -> ActionSpace:
        """Get action space object"""
        return self.action_space
    
    def get_token_name(self, token_id: int) -> str:
        """Get token name by ID"""
        return self.action_space.get_token_by_id(token_id)
    
    def get_token_id(self, token: int) -> str:
        """Get token name by ID"""
        return self.action_space.get_id_by_token(token)

    def get_state_debug_info(self, state: np.ndarray) -> Dict[str, Any]:
        """Get detailed debug information about current state"""
        pad_token_id = self.action_space.get_id_by_token('PAD')

        # Get non-PAD tokens
        non_pad_tokens = [int(t) for t in state if t != pad_token_id]
        token_names = [self.action_space.get_token_by_id(t) for t in non_pad_tokens]

        # Get payload
        payload = self.state_to_payload(state)

        return {
            'total_tokens': len(state),
            'non_pad_tokens': len(non_pad_tokens),
            'pad_tokens': len(state) - len(non_pad_tokens),
            'token_ids': non_pad_tokens,
            'token_names': token_names,
            'payload': payload,
            'payload_length': len(payload),
            'is_empty': len(non_pad_tokens) == 0,
            'is_full': pad_token_id not in state
        }
    
    def process_action(self, current_state: np.ndarray, action: int, processed_token: str = None) -> Dict[str, Any]:
        """
        Process an action (token selection) and return new state and info.

        Args:
            current_state: Current state array
            action: Token ID selected by agent
            processed_token: Token after bypass processing (if any)
        """
        # Use processed token if provided, otherwise get original token
        if processed_token is not None:
            final_token_text = processed_token
        else:
            final_token_text = self.payload_generator.token_to_text(action)

        # Append token to state
        new_state = self.append_token_to_state(current_state, action)

        # Build payload by concatenating all tokens
        payload = self._build_payload_from_state(new_state, final_token_text, action)

        # Get state information
        state_info = self.get_state_info(new_state)

        # Get token information
        token_name = self.get_token_name(action)

        return {
            'new_state': new_state,
            'token_id': action,
            'token_name': token_name,
            'original_token': self.payload_generator.token_to_text(action),
            'processed_token': final_token_text,
            'payload': payload,
            'is_complete': state_info['is_complete'],
            'is_full': state_info['is_full'],
            'payload_length': state_info['length']
        }

    def _build_payload_from_state(self, state: np.ndarray, latest_token: str, latest_token_id: int) -> str:
        """
        Build payload by concatenating tokens from state.

        Args:
            state: Current state array
            latest_token: The latest processed token text
            latest_token_id: The latest token ID
        """
        payload_parts = []
        pad_token_id = self.action_space.get_id_by_token('PAD')

        # Process all non-PAD tokens in state
        for i, token_id in enumerate(state):
            if token_id == pad_token_id:
                continue

            # Use the processed version for the latest token
            if i == len([t for t in state if t != pad_token_id]) - 1 and token_id == latest_token_id:
                token_text = latest_token
            else:
                token_text = self.payload_generator.token_to_text(int(token_id))

            if token_text and token_text not in ['PAD', 'END_TOKEN']:
                payload_parts.append(token_text)

        # Join tokens with appropriate spacing
        return self._join_tokens_intelligently(payload_parts)

    def _join_tokens_intelligently(self, tokens: List[str]) -> str:
        """
        Join tokens with intelligent spacing for SQL injection payloads.

        Args:
            tokens: List of token strings

        Returns:
            Joined payload string
        """
        if not tokens:
            return ""

        result = []

        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
                continue

            prev_token = tokens[i-1]

            # No space needed before/after certain characters
            no_space_before = ['(', ')', ',', ';', '.', '=', '!=', '<', '>', '<=', '>=']
            no_space_after = ['(', '.', '=', '!=', '<', '>', '<=', '>=']

            # Check if we need space
            need_space = True

            if token in no_space_before or prev_token in no_space_after:
                need_space = False

            # Special cases for quotes and comments
            if token in ["'", '"'] or prev_token in ["'", '"']:
                need_space = False

            if token.startswith('--') or token.startswith('/*'):
                need_space = True

            # Add token with or without space
            if need_space:
                result.append(' ' + token)
            else:
                result.append(token)

        return ''.join(result)
