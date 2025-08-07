"""
Rule engine for penetration testing guidelines and decision making
"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import re


class RuleType(Enum):
    """Types of rules in the rule engine."""
    CONDITION = "condition"
    ACTION = "action"
    VALIDATION = "validation"
    RECOMMENDATION = "recommendation"


class RuleOperator(Enum):
    """Operators for rule conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX_MATCH = "regex_match"
    IN_LIST = "in"
    NOT_IN_LIST = "not_in"


class Rule:
    """
    Represents a single rule in the rule engine.
    """
    
    def __init__(self, rule_id: str, rule_type: RuleType, conditions: List[Dict[str, Any]], 
                 actions: List[Dict[str, Any]], priority: int = 0, enabled: bool = True):
        """
        Initialize a rule.
        
        Args:
            rule_id: Unique identifier for the rule
            rule_type: Type of the rule
            conditions: List of conditions that must be met
            actions: List of actions to execute when conditions are met
            priority: Rule priority (higher number = higher priority)
            enabled: Whether the rule is enabled
        """
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.enabled = enabled
        self.execution_count = 0
        self.last_executed = None
    
    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate all conditions for this rule.
        
        Args:
            context: Context data to evaluate against
            
        Returns:
            True if all conditions are met
        """
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            if not self._evaluate_single_condition(condition, context):
                return False
        
        return True
    
    def _evaluate_single_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate a single condition.
        
        Args:
            condition: Condition to evaluate
            context: Context data
            
        Returns:
            True if condition is met
        """
        field = condition.get('field')
        operator = condition.get('operator')
        expected_value = condition.get('value')
        
        if field not in context:
            return False
        
        actual_value = context[field]
        
        try:
            if operator == RuleOperator.EQUALS.value:
                return actual_value == expected_value
            elif operator == RuleOperator.NOT_EQUALS.value:
                return actual_value != expected_value
            elif operator == RuleOperator.GREATER_THAN.value:
                return float(actual_value) > float(expected_value)
            elif operator == RuleOperator.LESS_THAN.value:
                return float(actual_value) < float(expected_value)
            elif operator == RuleOperator.GREATER_EQUAL.value:
                return float(actual_value) >= float(expected_value)
            elif operator == RuleOperator.LESS_EQUAL.value:
                return float(actual_value) <= float(expected_value)
            elif operator == RuleOperator.CONTAINS.value:
                return str(expected_value).lower() in str(actual_value).lower()
            elif operator == RuleOperator.NOT_CONTAINS.value:
                return str(expected_value).lower() not in str(actual_value).lower()
            elif operator == RuleOperator.REGEX_MATCH.value:
                return bool(re.search(str(expected_value), str(actual_value)))
            elif operator == RuleOperator.IN_LIST.value:
                return actual_value in expected_value
            elif operator == RuleOperator.NOT_IN_LIST.value:
                return actual_value not in expected_value
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    def execute_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute all actions for this rule.
        
        Args:
            context: Context data
            
        Returns:
            List of action results
        """
        results = []
        
        for action in self.actions:
            result = self._execute_single_action(action, context)
            results.append(result)
        
        self.execution_count += 1
        from datetime import datetime
        self.last_executed = datetime.now().isoformat()
        
        return results
    
    def _execute_single_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action.
        
        Args:
            action: Action to execute
            context: Context data
            
        Returns:
            Action result
        """
        action_type = action.get('type')
        action_data = action.get('data', {})
        
        result = {
            'rule_id': self.rule_id,
            'action_type': action_type,
            'success': True,
            'message': '',
            'data': {}
        }
        
        try:
            if action_type == 'recommend_payload':
                result['data'] = {
                    'recommended_payload': action_data.get('payload'),
                    'reason': action_data.get('reason'),
                    'confidence': action_data.get('confidence', 0.5)
                }
                result['message'] = f"Recommended payload: {action_data.get('payload')}"
            
            elif action_type == 'suggest_technique':
                result['data'] = {
                    'technique': action_data.get('technique'),
                    'description': action_data.get('description'),
                    'parameters': action_data.get('parameters', {})
                }
                result['message'] = f"Suggested technique: {action_data.get('technique')}"
            
            elif action_type == 'warn':
                result['data'] = {
                    'warning_message': action_data.get('message'),
                    'severity': action_data.get('severity', 'medium')
                }
                result['message'] = f"Warning: {action_data.get('message')}"
            
            elif action_type == 'modify_parameter':
                result['data'] = {
                    'parameter': action_data.get('parameter'),
                    'new_value': action_data.get('value'),
                    'operation': action_data.get('operation', 'set')
                }
                result['message'] = f"Modified parameter: {action_data.get('parameter')}"
            
            elif action_type == 'set_flag':
                result['data'] = {
                    'flag_name': action_data.get('flag'),
                    'flag_value': action_data.get('value', True)
                }
                result['message'] = f"Set flag: {action_data.get('flag')}"
            
            else:
                result['success'] = False
                result['message'] = f"Unknown action type: {action_type}"
        
        except Exception as e:
            result['success'] = False
            result['message'] = f"Error executing action: {str(e)}"
        
        return result


class RuleEngine:
    """
    Rule engine for penetration testing decision making and guidance.
    """
    
    def __init__(self):
        """
        Initialize the rule engine.
        """
        self.rules = {}
        self.rule_sets = {}
        self.execution_history = []
        self.context_validators = {}
        
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """
        Initialize default rules for SQL injection testing.
        """
        # Rule for recommending basic payloads when starting
        self.add_rule(Rule(
            rule_id="start_with_basic_payloads",
            rule_type=RuleType.RECOMMENDATION,
            conditions=[
                {'field': 'attempt_count', 'operator': '==', 'value': 0}
            ],
            actions=[
                {
                    'type': 'recommend_payload',
                    'data': {
                        'payload': "' OR '1'='1",
                        'reason': 'Basic boolean test for initial detection',
                        'confidence': 0.8
                    }
                }
            ],
            priority=10
        ))
        
        # Rule for detecting WAF and suggesting bypass
        self.add_rule(Rule(
            rule_id="waf_bypass_suggestion",
            rule_type=RuleType.RECOMMENDATION,
            conditions=[
                {'field': 'waf_detected', 'operator': '==', 'value': True},
                {'field': 'waf_triggered', 'operator': '==', 'value': True}
            ],
            actions=[
                {
                    'type': 'suggest_technique',
                    'data': {
                        'technique': 'encoding_bypass',
                        'description': 'Try URL encoding to bypass WAF',
                        'parameters': {'encoding_type': 'url'}
                    }
                }
            ],
            priority=8
        ))
        
        # Rule for escalating to time-based when blind injection suspected
        self.add_rule(Rule(
            rule_id="escalate_to_time_based",
            rule_type=RuleType.RECOMMENDATION,
            conditions=[
                {'field': 'blind_injection_possible', 'operator': '==', 'value': True},
                {'field': 'error_detected', 'operator': '==', 'value': False}
            ],
            actions=[
                {
                    'type': 'recommend_payload',
                    'data': {
                        'payload': "' AND SLEEP(5)--",
                        'reason': 'Time-based testing for blind injection confirmation',
                        'confidence': 0.7
                    }
                }
            ],
            priority=7
        ))
        
        # Rule for warning about excessive attempts
        self.add_rule(Rule(
            rule_id="excessive_attempts_warning",
            rule_type=RuleType.VALIDATION,
            conditions=[
                {'field': 'attempt_count', 'operator': '>', 'value': 50}
            ],
            actions=[
                {
                    'type': 'warn',
                    'data': {
                        'message': 'High number of attempts may trigger detection systems',
                        'severity': 'high'
                    }
                }
            ],
            priority=9
        ))
        
        # Rule for success detection
        self.add_rule(Rule(
            rule_id="injection_success_detected",
            rule_type=RuleType.ACTION,
            conditions=[
                {'field': 'injection_detected', 'operator': '==', 'value': True}
            ],
            actions=[
                {
                    'type': 'set_flag',
                    'data': {
                        'flag': 'exploitation_successful',
                        'value': True
                    }
                },
                {
                    'type': 'suggest_technique',
                    'data': {
                        'technique': 'data_extraction',
                        'description': 'Proceed with data extraction techniques',
                        'parameters': {'extraction_method': 'union_based'}
                    }
                }
            ],
            priority=10
        ))
        
        # Rule for database fingerprinting
        self.add_rule(Rule(
            rule_id="database_fingerprinting",
            rule_type=RuleType.RECOMMENDATION,
            conditions=[
                {'field': 'injection_detected', 'operator': '==', 'value': True},
                {'field': 'database_type', 'operator': '==', 'value': 'unknown'}
            ],
            actions=[
                {
                    'type': 'recommend_payload',
                    'data': {
                        'payload': "' UNION SELECT @@version,2,3--",
                        'reason': 'Database version detection',
                        'confidence': 0.9
                    }
                }
            ],
            priority=6
        ))
    
    def add_rule(self, rule: Rule):
        """
        Add a rule to the engine.
        
        Args:
            rule: Rule to add
        """
        self.rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the engine.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if rule was removed
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a specific rule.
        
        Args:
            rule_id: ID of rule to enable
            
        Returns:
            True if rule was enabled
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable a specific rule.
        
        Args:
            rule_id: ID of rule to disable
            
        Returns:
            True if rule was disabled
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            return True
        return False
    
    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all rules against the given context.
        
        Args:
            context: Context data to evaluate
            
        Returns:
            List of action results from triggered rules
        """
        triggered_rules = []
        
        # Find all rules with satisfied conditions
        for rule in self.rules.values():
            if rule.evaluate_conditions(context):
                triggered_rules.append(rule)
        
        # Sort by priority (higher priority first)
        triggered_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Execute actions
        all_results = []
        for rule in triggered_rules:
            results = rule.execute_actions(context)
            all_results.extend(results)
            
            # Record execution
            self.execution_history.append({
                'rule_id': rule.rule_id,
                'context_snapshot': context.copy(),
                'results': results,
                'timestamp': rule.last_executed
            })
        
        return all_results
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get recommendations based on current context.
        
        Args:
            context: Current context
            
        Returns:
            List of recommendations
        """
        results = self.evaluate(context)
        recommendations = []
        
        for result in results:
            if result.get('action_type') in ['recommend_payload', 'suggest_technique']:
                recommendations.append(result)
        
        return recommendations
    
    def validate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the context and return warnings/errors.
        
        Args:
            context: Context to validate
            
        Returns:
            Validation results
        """
        results = self.evaluate(context)
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        for result in results:
            if result.get('action_type') == 'warn':
                severity = result.get('data', {}).get('severity', 'medium')
                warning = {
                    'message': result.get('message'),
                    'severity': severity,
                    'rule_id': result.get('rule_id')
                }
                
                if severity == 'high':
                    validation_results['errors'].append(warning)
                    validation_results['valid'] = False
                else:
                    validation_results['warnings'].append(warning)
        
        return validation_results
    
    def create_rule_set(self, rule_set_id: str, rule_ids: List[str], 
                       description: str = '') -> bool:
        """
        Create a rule set for grouped rule management.
        
        Args:
            rule_set_id: Identifier for the rule set
            rule_ids: List of rule IDs to include
            description: Description of the rule set
            
        Returns:
            True if rule set was created
        """
        # Validate that all rule IDs exist
        valid_rule_ids = [rid for rid in rule_ids if rid in self.rules]
        
        if len(valid_rule_ids) != len(rule_ids):
            return False
        
        self.rule_sets[rule_set_id] = {
            'rule_ids': valid_rule_ids,
            'description': description,
            'created': None  # Would use datetime.now() in real implementation
        }
        
        return True
    
    def enable_rule_set(self, rule_set_id: str) -> bool:
        """
        Enable all rules in a rule set.
        
        Args:
            rule_set_id: Rule set identifier
            
        Returns:
            True if rule set was enabled
        """
        if rule_set_id not in self.rule_sets:
            return False
        
        rule_ids = self.rule_sets[rule_set_id]['rule_ids']
        for rule_id in rule_ids:
            self.enable_rule(rule_id)
        
        return True
    
    def disable_rule_set(self, rule_set_id: str) -> bool:
        """
        Disable all rules in a rule set.
        
        Args:
            rule_set_id: Rule set identifier
            
        Returns:
            True if rule set was disabled
        """
        if rule_set_id not in self.rule_sets:
            return False
        
        rule_ids = self.rule_sets[rule_set_id]['rule_ids']
        for rule_id in rule_ids:
            self.disable_rule(rule_id)
        
        return True
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about rule execution.
        
        Returns:
            Rule execution statistics
        """
        stats = {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
            'disabled_rules': len([r for r in self.rules.values() if not r.enabled]),
            'total_executions': len(self.execution_history),
            'rule_execution_counts': {},
            'most_triggered_rule': None,
            'rule_types': {}
        }
        
        # Count executions per rule
        for rule_id, rule in self.rules.items():
            stats['rule_execution_counts'][rule_id] = rule.execution_count
        
        # Find most triggered rule
        if stats['rule_execution_counts']:
            most_triggered = max(stats['rule_execution_counts'], 
                               key=stats['rule_execution_counts'].get)
            stats['most_triggered_rule'] = most_triggered
        
        # Count by rule type
        for rule in self.rules.values():
            rule_type = rule.rule_type.value
            stats['rule_types'][rule_type] = stats['rule_types'].get(rule_type, 0) + 1
        
        return stats
    
    def export_rules(self, format: str = 'json') -> str:
        """
        Export rules in specified format.
        
        Args:
            format: Export format (json, yaml, etc.)
            
        Returns:
            Exported rules as string
        """
        rules_data = {}
        
        for rule_id, rule in self.rules.items():
            rules_data[rule_id] = {
                'rule_type': rule.rule_type.value,
                'conditions': rule.conditions,
                'actions': rule.actions,
                'priority': rule.priority,
                'enabled': rule.enabled,
                'execution_count': rule.execution_count
            }
        
        if format == 'json':
            import json
            return json.dumps(rules_data, indent=2)
        else:
            return str(rules_data)
    
    def clear_execution_history(self):
        """
        Clear the rule execution history.
        """
        self.execution_history.clear()
        for rule in self.rules.values():
            rule.execution_count = 0
            rule.last_executed = None
