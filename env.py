"""
Environment Module for SQL Injection RL Agent
Handles HTTP requests and reward calculation
"""

import requests
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from urllib.parse import urljoin, quote
import numpy as np

from gen_action import GenAction
from bypass_waf import BypassWAF
from simple_state import SimpleStateManager


class SQLiEnvironment:
    """SQL Injection testing environment for RL agent"""
    
    def __init__(self, target_url: str = "http://localhost:8080/vuln",
                 parameter: str = "id", method: str = "GET",
                 injection_point: str = "1", max_steps: int = 50, timeout: int = 10,
                 blocked_keywords: Optional[Union[List[str], str]] = None):

        self.target_url = target_url
        self.parameter = parameter
        self.method = method.upper()
        self.injection_point = injection_point
        self.max_steps = max_steps
        self.timeout = timeout

        # Initialize modules
        self.gen_action = GenAction()
        self.bypass_waf = BypassWAF(blocked_keywords=blocked_keywords)
        self.state_manager = SimpleStateManager()

        # Environment state
        self.current_state = None
        self.current_payload = ""
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
            baseline_url = self._build_injection_url("")
            response = self._send_request_to_url(baseline_url)
            self.baseline_response = response
            print(f"✅ Baseline established: Status {response['status_code']}, Length {response['content_length']}")
        except Exception as e:
            print(f"⚠️ Could not establish baseline: {e}")
            self.baseline_response = {'status_code': 200, 'content': '', 'content_length': 0, 'response_time': 1.0}
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.step_count = 0
        self.current_payload = ""
        self.episode_history = []
        self.state_manager.reset()

        # Build initial state using SimpleStateManager
        self.current_state = self.state_manager.build_state(
            current_payload=self.current_payload,
            step_count=self.step_count,
            max_steps=self.max_steps
        )

        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Get original token
        original_token = self.gen_action.payload_generator.token_to_text(action)
        token_name = self.gen_action.get_token_name(action)
        
        


        # Check if token should be bypassed
        should_bypass = self.bypass_waf.should_bypass_token(original_token)
        processed_token = original_token
        bypass_info = None

        # Apply bypass if needed
        if should_bypass:
            for method in self.bypass_waf.get_available_methods():
                bypass_result = self.bypass_waf.apply_bypass_to_token(original_token, method=method)
                if bypass_result['success']:
                    processed_token = bypass_result['bypassed']
                    bypass_info = bypass_result
                    break

        # Update payload by appending processed token
        if self.current_payload:
            self.current_payload += processed_token
        else:
            self.current_payload = processed_token
        
        # Build final URL
        final_url = self._build_injection_url(self.current_payload)

        # Send HTTP request
        response = self._send_request_to_url(final_url)

        # Check if response indicates blocking
        is_blocked = self.bypass_waf.is_likely_blocked(
            response['status_code'],
            response['content'],
            response['response_time']
        )

        # Detect SQL errors and calculate reward
        error_info = self._detect_sql_error(response['content'])
        reward = self._calculate_reward(response, self.current_payload, error_info)

        # Build new state using SimpleStateManager
        self.current_state = self.state_manager.build_state(
            current_payload=self.current_payload,
            response=response,
            is_blocked=is_blocked,
            bypass_applied=should_bypass,
            bypass_method=bypass_info['method'] if bypass_info else None,
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=reward
        )

        # Check if episode is done
        done = self._is_episode_done(response)
        
        # Prepare info
        info = {
            'payload': self.current_payload,
            'original_token': original_token,
            'processed_token': processed_token,
            'token_name': token_name,
            'final_url': final_url,
            'response_status': response['status_code'],
            'response_length': response['content_length'],
            'response_time': response['response_time'],
            'is_blocked': is_blocked,
            'bypass_applied': should_bypass,
            'bypass_method': bypass_info['method'] if bypass_info else None,
            'step_count': self.step_count,
            'sqli_detected': self._detect_sqli_success(response),
            'error_detected': error_info['has_error'],
            'error_info': error_info,
            'is_complete': len(self.current_payload) >= 500,  # Simple completion check
            'is_full': len(self.current_payload) >= 500
        }
        
        # Store in history
        self.episode_history.append({
            'action': action,
            'payload': self.current_payload,
            'response': response,
            'reward': reward,
            'info': info
        })
        
        if info['sqli_detected']:
            with open("sqli_success_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Payload: {self.current_payload}\n")
                f.write(f"Bypass method: {info.get('bypass_method', 'None')}\n")
                f.write(f"Response (preview): {response.get('content', '')[:1000]}\n")
                f.write("="*60 + "\n")
        
        print("Action selected: ", token_name)
        print("Current payload", self.current_payload)
        print("Final url", final_url)

        return self.current_state, reward, done, info
    
    def _build_injection_url(self, payload: str) -> str:
        """Build final URL by injecting payload with proper SQL syntax"""
        if not payload.strip():
            if self.method == "GET":
                return f"{self.target_url}?{self.parameter}={self.injection_point}"
            else:
                return self.target_url

        # Build injection value with proper SQL syntax
        injection_value = self._build_injection_value(payload)

        if self.method == "GET":
            if '?' in self.target_url:
                if f"{self.parameter}=" in self.target_url:
                    import re
                    # Escape special regex characters in injection_point
                    escaped_injection_point = re.escape(str(self.injection_point))
                    pattern = f"({self.parameter}={escaped_injection_point})"
                    replacement = f"{self.parameter}={injection_value}"
                    try:
                        final_url = re.sub(pattern, replacement, self.target_url)
                    except re.error as e:
                        # Fallback: simple string replacement
                        old_param = f"{self.parameter}={self.injection_point}"
                        new_param = f"{self.parameter}={injection_value}"
                        final_url = self.target_url.replace(old_param, new_param)
                else:
                    final_url = f"{self.target_url}&{self.parameter}={injection_value}"
            else:
                final_url = f"{self.target_url}?{self.parameter}={injection_value}"
        else:
            final_url = self.target_url

        return final_url

    def _build_injection_value(self, payload: str) -> str:
        """Build injection value with proper SQL syntax"""
        if not payload.strip():
            return self.injection_point
        return f"{self.injection_point}{payload}"

        # # Determine injection context based on payload content
        # payload_upper = payload.upper()

        # # For UNION-based injections
        # if 'UNION' in payload_upper and 'SELECT' in payload_upper:
        #     return f"{self.injection_point} {payload}"

        # # For boolean-based injections (AND/OR)
        # elif any(keyword in payload_upper for keyword in ['AND', 'OR']) and not payload.startswith(('AND', 'OR')):
        #     return f"{self.injection_point} {payload}"

        # # For error-based injections
        # elif any(func in payload_upper for func in ['EXTRACTVALUE', 'UPDATEXML', 'XMLTYPE']):
        #     return f"{self.injection_point} AND {payload}"

        # # For time-based injections
        # elif any(func in payload_upper for func in ['SLEEP', 'WAITFOR', 'BENCHMARK']):
        #     return f"{self.injection_point} AND {payload}"

        # # For string-based injections (quotes)
        # elif "'" in payload:
        #     # Try different quote contexts
        #     if payload.startswith("'"):
        #         return f"{self.injection_point}{payload}"  # ?id=1'...
        #     else:
        #         return f"{self.injection_point}' {payload}"  # ?id=1' ...

        # # For comment-based injections
        # elif payload.startswith('--') or payload.startswith('/*'):
        #     return f"{self.injection_point} {payload}"

        # # For numeric injections (default)
        # else:
        #     # Add space for readability
        
    
    def _send_request_to_url(self, url: str) -> Dict[str, Any]:
        """Send HTTP request to URL"""
        start_time = time.time()
        
        try:
            if self.method == "GET":
                response = self.session.get(url, timeout=self.timeout, allow_redirects=False)
            else:
                if '?' in url and f"{self.parameter}=" in url:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url)
                    params = urllib.parse.parse_qs(parsed.query)
                    payload_value = params.get(self.parameter, [''])[0]
                    data = {self.parameter: payload_value}
                    
                else:
                    data = {self.parameter: self.injection_point}
                print(f"[DEBUG] Sending POST request: {self.target_url} | data={data}")
                response = self.session.post(self.target_url, data=data, timeout=self.timeout, allow_redirects=False)
            
            response_time = time.time() - start_time
            #print("response content:", response.text)
            return {
                'status_code': response.status_code,
                'content': response.text,
                'content_length': len(response.text),
                'response_time': response_time,
                'headers': dict(response.headers),
                'url': response.url
            }
            
        except requests.exceptions.Timeout:
            return {
                'status_code': 0,
                'content': '',
                'content_length': 0,
                'response_time': time.time() - start_time,
                'headers': {},
                'url': url,
                'error': 'timeout'
            }
        except Exception as e:
            return {
                'status_code': 0,
                'content': '',
                'content_length': 0,
                'response_time': time.time() - start_time,
                'headers': {},
                'url': url,
                'error': str(e)
            }
    
    def _detect_sql_error(self, content: str) -> Dict[str, Any]:
        """Detect SQL error messages and extract information"""
        content_lower = content.lower()
        
        error_patterns = {
            'mysql': ['unknown column', 'mysql_fetch_array', 'mysql_num_rows', 'you have an error in your sql syntax', 'different number of columns'],
            'postgresql': ['postgresql', 'pg_query', 'column does not exist', 'relation does not exist'],
            'oracle': ['ora-', 'oracle', 'oci_execute'],
            'mssql': ['microsoft ole db provider', 'unclosed quotation mark', 'syntax error'],
            'sqlite': ['sqlite', 'no such column', 'no such table']
        }
        
        detected_errors = []
        database_type = 'unknown'
        
        for db_type, patterns in error_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    detected_errors.append(pattern)
                    database_type = db_type
                    break
        
        # Extract column names
        column_matches = re.findall(r"unknown column ['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
        table_matches = re.findall(r"table ['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
        
        return {
            'has_error': len(detected_errors) > 0,
            'error_count': len(detected_errors),
            'detected_patterns': detected_errors,
            'database_type': database_type,
            'column_names': column_matches,
            'table_names': table_matches
        }
    
    def _calculate_reward(self, response: Dict[str, Any], payload: str, error_info: Dict[str, Any]) -> float:
        """Calculate reward based on response"""
        status_code = response.get('status_code', 0)
        content = response.get('content', '')
        response_time = response.get('response_time', 0)
        
        # High reward for SQL injection success
        if self._detect_sqli_success(response):
            bonus = 0
            if re.search(r"flag\{.*?\}|ctf\{.*?\}|root@localhost|version\(|user\(", content, re.I):
                bonus = 1.0
            return 2.0 + bonus
        
        # Enhanced reward for SQL errors
        if error_info['has_error']:
            base_reward = 0.5
            if error_info['database_type'] != 'unknown':
                base_reward += 0.2
            if error_info['column_names']:
                base_reward += 0.3
            if error_info['table_names']:
                base_reward += 0.2
            return min(base_reward, 0.9)
            # Reward khi bypass thành công
        if response.get('bypass_applied', False) and not response.get('is_blocked', False):
            return 0.5
        
        # Negative rewards
        if status_code in [403, 406, 429, 501, 503]:
            return -1.0
        if self.bypass_waf.is_likely_blocked(status_code, content, response_time):
            return -0.5
        if status_code == 0:
            return -0.3
        
        # Small positive for different responses
        if self._is_response_different(response):
            return 0.1
        
        if len(payload) > 400:
            return -0.2


        return 0.0
    
    def _detect_sqli_success(self, response: Dict[str, Any]) -> bool:
        """Detect likely SQL injection success"""
        content = response.get('content', '').lower()
        response_time = response.get('response_time', 0)
        
        # Time-based detection
        if response_time > 4.5:
            return True
        
        # Extract data
        success_patterns = [
            r"zixem@localhost$", r"8.0.36",
            # r"root@localhost", r"information_schema", r"mysql_fetch_array",
            # r"mysql_num_rows", r"flag\{.*?\}", r"ctf\{.*?\}",
            # r"version\(", r"user\(", r"admin", r"select .* from", r"union select",
            # r"table .* does not exist", r"column .* does not exist", r"8\.0\.\d+", r"5\.\d+\.\d+"
        ]
        for pattern in success_patterns:
            if re.search(pattern, content, re.I):
                return True

        return False
    
    def _is_response_different(self, response: Dict[str, Any]) -> bool:
        """Check if response differs from baseline"""
        if not self.baseline_response:
            return False
        
        status_diff = response.get('status_code', 0) != self.baseline_response.get('status_code', 0)
        baseline_length = self.baseline_response.get('content_length', 0)
        current_length = response.get('content_length', 0)
        length_diff = baseline_length > 0 and abs(current_length - baseline_length) > baseline_length * 0.1
        
        return status_diff or length_diff
    
    def _is_episode_done(self, response: Dict[str, Any]) -> bool:
        """Determine if episode should end"""
        return (self.step_count >= self.max_steps or
                self._detect_sqli_success(response) or
                len(self.current_payload) >= 500)
    
    def get_state_size(self) -> int:
        """Get state size"""
        return self.state_manager.get_state_size()

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
        
        return {
            'total_steps': len(self.episode_history),
            'total_reward': total_reward,
            'average_reward': total_reward / len(self.episode_history),
            'sqli_detected': sqli_detected,
            'errors_detected': errors_detected,
            'final_payload': self.episode_history[-1]['payload'] if self.episode_history else '',
            'success_rate': 1.0 if sqli_detected else 0.0
        }
