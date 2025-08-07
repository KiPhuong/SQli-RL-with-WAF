"""
Environment Module for SQL Injection RL Agent
Handles HTTP requests and reward calculation
"""

import requests
import time
import re
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urljoin, quote
import numpy as np

from gen_action import GenAction
from bypass_waf import BypassWAF


class SQLiEnvironment:
    """SQL Injection testing environment for RL agent"""
    
    def __init__(self, target_url: str = "http://localhost:8080/vuln", 
                 parameter: str = "id", method: str = "GET", 
                 max_steps: int = 50, timeout: int = 10):
        
        self.target_url = target_url
        self.parameter = parameter
        self.method = method.upper()
        self.max_steps = max_steps
        self.timeout = timeout
        
        # Initialize modules
        self.gen_action = GenAction()
        self.bypass_waf = BypassWAF()
        
        # Environment state
        self.current_state = None
        self.step_count = 0
        self.episode_history = []
        self.baseline_response = None
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize baseline
        self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish baseline response for comparison"""
        try:
            baseline_payload = "1"  # Simple baseline
            response = self._send_request(baseline_payload)
            self.baseline_response = response
            print(f"Baseline established: Status {response['status_code']}, "
                  f"Length {response['content_length']}")
        except Exception as e:
            print(f"Warning: Could not establish baseline: {e}")
            self.baseline_response = {
                'status_code': 200,
                'content': '',
                'content_length': 0,
                'response_time': 1.0
            }
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.current_state = self.gen_action.create_initial_state()
        self.step_count = 0
        self.episode_history = []
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Token ID selected by agent
            
        Returns:
            next_state, reward, done, info
        """
        self.step_count += 1
        
        # Process action through gen-action module
        action_result = self.gen_action.process_action(self.current_state, action)
        new_state = action_result['new_state']
        payload = action_result['payload']
        
        # Check if token should be bypassed
        token_name = action_result['token_name']
        should_bypass = self.bypass_waf.should_bypass_token(token_name)
        
        # Apply bypass if needed
        if should_bypass:
            payload = self.bypass_waf.apply_bypass(payload)
        
        # Send HTTP request
        response = self._send_request(payload)
        
        # Check if response indicates blocking
        is_blocked = self.bypass_waf.is_likely_blocked(
            response['status_code'], 
            response['content'], 
            response['response_time']
        )
        
        # Apply additional bypass if blocked
        if is_blocked and not should_bypass:
            payload = self.bypass_waf.apply_bypass(payload)
            response = self._send_request(payload)
        
        # Calculate reward
        reward = self._calculate_reward(response, payload)
        
        # Check if episode is done
        done = self._is_episode_done(response, action_result)
        
        # Update state
        self.current_state = new_state
        
        # Prepare info
        info = {
            'payload': payload,
            'token_name': token_name,
            'response_status': response['status_code'],
            'response_length': response['content_length'],
            'response_time': response['response_time'],
            'is_blocked': is_blocked,
            'bypass_applied': should_bypass,
            'step_count': self.step_count,
            'sqli_detected': self._detect_sqli_success(response),
            'error_detected': self._detect_sql_error(response['content'])
        }
        
        # Store in history
        self.episode_history.append({
            'action': action,
            'payload': payload,
            'response': response,
            'reward': reward,
            'info': info
        })
        
        return new_state, reward, done, info
    
    def _send_request(self, payload: str) -> Dict[str, Any]:
        """Send HTTP request with payload"""
        start_time = time.time()
        
        try:
            if self.method == "GET":
                params = {self.parameter: payload}
                response = self.session.get(
                    self.target_url, 
                    params=params, 
                    timeout=self.timeout,
                    allow_redirects=False
                )
            else:  # POST
                data = {self.parameter: payload}
                response = self.session.post(
                    self.target_url, 
                    data=data, 
                    timeout=self.timeout,
                    allow_redirects=False
                )
            
            response_time = time.time() - start_time
            
            return {
                'status_code': response.status_code,
                'content': response.text,
                'content_length': len(response.text),
                'response_time': response_time,
                'headers': dict(response.headers),
                'url': response.url
            }
            
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            return {
                'status_code': 0,
                'content': '',
                'content_length': 0,
                'response_time': response_time,
                'headers': {},
                'url': self.target_url,
                'error': 'timeout'
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'status_code': 0,
                'content': '',
                'content_length': 0,
                'response_time': response_time,
                'headers': {},
                'url': self.target_url,
                'error': str(e)
            }
    
    def _calculate_reward(self, response: Dict[str, Any], payload: str) -> float:
        """Calculate reward based on response"""
        status_code = response['status_code']
        content = response['content']
        response_time = response['response_time']
        
        # High positive reward for likely SQL injection success
        if self._detect_sqli_success(response):
            return 1.0
        
        # Positive reward for SQL errors (indicates injection potential)
        if self._detect_sql_error(content):
            return 0.5
        
        # Negative reward for clear blocking
        if status_code in [403, 406, 429, 501, 503]:
            return -1.0
        
        # Negative reward for WAF detection
        if self.bypass_waf.is_likely_blocked(status_code, content, response_time):
            return -0.5
        
        # Small positive reward for different responses (exploration)
        if self._is_response_different(response):
            return 0.1
        
        # Small negative reward for timeouts
        if status_code == 0:
            return -0.3
        
        # Neutral reward for normal responses
        return 0.0
    
    def _detect_sqli_success(self, response: Dict[str, Any]) -> bool:
        """Detect likely SQL injection success"""
        content = response['content'].lower()
        status_code = response['status_code']
        response_time = response['response_time']
        
        # Time-based detection (significant delay)
        if response_time > 5.0:
            return True
        
        # Error-based detection
        sql_error_patterns = [
            'mysql_fetch_array', 'mysql_num_rows', 'mysql_error',
            'ora-01756', 'ora-00933', 'microsoft ole db provider',
            'unclosed quotation mark', 'quoted string not properly terminated',
            'syntax error', 'unexpected end of sql command',
            'warning: mysql', 'warning: pg_', 'warning: oci_',
            'microsoft jet database', 'odbc microsoft access'
        ]
        
        for pattern in sql_error_patterns:
            if pattern in content:
                return True
        
        # Union-based detection (additional columns in response)
        if self.baseline_response and status_code == 200:
            baseline_length = self.baseline_response['content_length']
            current_length = response['content_length']
            
            # Significant increase in response length
            if current_length > baseline_length * 1.5:
                return True
        
        return False
    
    def _detect_sql_error(self, content: str) -> bool:
        """Detect SQL error messages"""
        content_lower = content.lower()
        error_indicators = [
            'sql syntax', 'mysql', 'postgresql', 'oracle', 'mssql',
            'syntax error', 'unexpected token', 'near', 'column',
            'table', 'database', 'query', 'statement'
        ]
        
        return any(indicator in content_lower for indicator in error_indicators)
    
    def _is_response_different(self, response: Dict[str, Any]) -> bool:
        """Check if response is different from baseline"""
        if not self.baseline_response:
            return False
        
        # Compare status codes
        if response['status_code'] != self.baseline_response['status_code']:
            return True
        
        # Compare content lengths (with tolerance)
        baseline_length = self.baseline_response['content_length']
        current_length = response['content_length']
        
        if abs(current_length - baseline_length) > baseline_length * 0.1:
            return True
        
        return False
    
    def _is_episode_done(self, response: Dict[str, Any], action_result: Dict[str, Any]) -> bool:
        """Determine if episode should end"""
        # End if maximum steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # End if SQL injection success detected
        if self._detect_sqli_success(response):
            return True
        
        # End if END_TOKEN is used
        if action_result['is_complete']:
            return True
        
        # End if state is full
        if action_result['is_full']:
            return True
        
        return False
    
    def get_state_size(self) -> int:
        """Get state size"""
        return self.gen_action.state_length
    
    def get_action_size(self) -> int:
        """Get action space size"""
        return self.gen_action.get_vocab_size()
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode"""
        if not self.episode_history:
            return {}
        
        total_reward = sum(step['reward'] for step in self.episode_history)
        sqli_detected = any(step['info']['sqli_detected'] for step in self.episode_history)
        errors_detected = sum(1 for step in self.episode_history if step['info']['error_detected'])
        blocks_encountered = sum(1 for step in self.episode_history if step['info']['is_blocked'])
        
        return {
            'total_steps': len(self.episode_history),
            'total_reward': total_reward,
            'average_reward': total_reward / len(self.episode_history),
            'sqli_detected': sqli_detected,
            'errors_detected': errors_detected,
            'blocks_encountered': blocks_encountered,
            'final_payload': self.episode_history[-1]['payload'] if self.episode_history else '',
            'success_rate': 1.0 if sqli_detected else 0.0
        }
