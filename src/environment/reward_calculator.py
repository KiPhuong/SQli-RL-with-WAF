"""
Reward calculation system for the SQL injection RL agent
"""

import numpy as np
from typing import Dict, Any, List
import math


class RewardCalculator:
    """
    Calculates rewards for the RL agent based on penetration testing outcomes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward calculator.
        
        Args:
            config: Reward configuration dictionary
        """
        self.config = config
        
        # Reward weights
        self.injection_success_reward = config.get('injection_success_reward', 100.0)
        self.error_discovery_reward = config.get('error_discovery_reward', 50.0)
        self.blind_injection_reward = config.get('blind_injection_reward', 75.0)
        self.waf_bypass_reward = config.get('waf_bypass_reward', 30.0)
        self.exploration_reward = config.get('exploration_reward', 5.0)
        self.efficiency_bonus = config.get('efficiency_bonus', 20.0)
        
        # Penalties
        self.waf_trigger_penalty = config.get('waf_trigger_penalty', -10.0)
        self.failed_attempt_penalty = config.get('failed_attempt_penalty', -1.0)
        self.repetitive_action_penalty = config.get('repetitive_action_penalty', -5.0)
        self.time_penalty = config.get('time_penalty', -0.1)
        
        # Tracking variables
        self.previous_discoveries = set()
        self.action_frequency = {}
        self.attempt_count = 0
        self.successful_injections = 0
    
    def calculate_reward(self, state: Dict[str, Any], action: int, 
                        response: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """
        Calculate reward for the current action and outcome.
        
        Args:
            state: Current environment state
            action: Action taken
            response: HTTP response received
            analysis: Response analysis results
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        self.attempt_count += 1
        
        # Primary success rewards
        reward += self._calculate_success_rewards(analysis, state)
        
        # Discovery rewards
        reward += self._calculate_discovery_rewards(analysis, response)
        
        # WAF-related rewards/penalties
        reward += self._calculate_waf_rewards(analysis, state)
        
        # Efficiency rewards
        reward += self._calculate_efficiency_rewards(state, action)
        
        # Exploration rewards
        reward += self._calculate_exploration_rewards(action, analysis)
        
        # Penalties
        reward += self._calculate_penalties(analysis, action, response)
        
        # Bonus rewards
        reward += self._calculate_bonus_rewards(state, analysis)
        
        # Update tracking variables
        self._update_tracking(action, analysis)
        
        return reward
    
    def _calculate_success_rewards(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Calculate rewards for successful injections.
        
        Args:
            analysis: Response analysis results
            state: Current state
            
        Returns:
            Success reward
        """
        reward = 0.0
        
        # Direct injection success
        if analysis.get('injection_detected', False):
            reward += self.injection_success_reward
            self.successful_injections += 1
            
            # Bonus for early success
            if self.attempt_count <= 10:
                reward += self.efficiency_bonus * (11 - self.attempt_count) / 10
        
        # Blind injection potential
        elif analysis.get('blind_injection_possible', False):
            reward += self.blind_injection_reward * 0.5  # Partial reward
        
        # Error-based injection
        if analysis.get('error_detected', False):
            reward += self.error_discovery_reward
        
        return reward
    
    def _calculate_discovery_rewards(self, analysis: Dict[str, Any], response: Dict[str, Any]) -> float:
        """
        Calculate rewards for discovering new information.
        
        Args:
            analysis: Response analysis results
            response: HTTP response
            
        Returns:
            Discovery reward
        """
        reward = 0.0
        
        # New error patterns discovered
        content = response.get('content', '').lower()
        error_patterns = [
            'mysql', 'oracle', 'postgresql', 'sql server', 'sqlite',
            'syntax error', 'column', 'table', 'database'
        ]
        
        for pattern in error_patterns:
            if pattern in content and pattern not in self.previous_discoveries:
                reward += 10.0
                self.previous_discoveries.add(pattern)
        
        # Database fingerprinting
        db_indicators = {
            'mysql': ['mysql_fetch_array', 'mysql_num_rows'],
            'oracle': ['ORA-', 'ORACLE'],
            'postgresql': ['PostgreSQL', 'pg_'],
            'mssql': ['Microsoft SQL Server', 'MSSQL']
        }
        
        for db_type, indicators in db_indicators.items():
            if any(indicator.lower() in content for indicator in indicators):
                if db_type not in self.previous_discoveries:
                    reward += 15.0
                    self.previous_discoveries.add(db_type)
        
        return reward
    
    def _calculate_waf_rewards(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Calculate WAF-related rewards and penalties.
        
        Args:
            analysis: Response analysis results
            state: Current state
            
        Returns:
            WAF-related reward/penalty
        """
        reward = 0.0
        
        # WAF bypass success
        if state.get('waf_detected', False) and analysis.get('injection_detected', False):
            reward += self.waf_bypass_reward
        
        # WAF trigger penalty
        if analysis.get('waf_triggered', False):
            reward += self.waf_trigger_penalty
            
            # Escalating penalty for repeated WAF triggers
            recent_triggers = getattr(self, 'recent_waf_triggers', [])
            recent_triggers.append(True)
            if len(recent_triggers) > 5:
                recent_triggers = recent_triggers[-5:]
            
            trigger_rate = sum(recent_triggers) / len(recent_triggers)
            if trigger_rate > 0.8:  # More than 80% recent triggers
                reward += self.waf_trigger_penalty * 2
            
            self.recent_waf_triggers = recent_triggers
        else:
            # Reset trigger tracking if no trigger
            recent_triggers = getattr(self, 'recent_waf_triggers', [])
            recent_triggers.append(False)
            if len(recent_triggers) > 5:
                recent_triggers = recent_triggers[-5:]
            self.recent_waf_triggers = recent_triggers
        
        return reward
    
    def _calculate_efficiency_rewards(self, state: Dict[str, Any], action: int) -> float:
        """
        Calculate efficiency-based rewards.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Efficiency reward
        """
        reward = 0.0
        
        # Success rate bonus
        if self.attempt_count > 0:
            success_rate = self.successful_injections / self.attempt_count
            if success_rate > 0.1:  # Above 10% success rate
                reward += self.efficiency_bonus * success_rate
        
        # Quick discovery bonus (inverse relationship with attempt count)
        if self.successful_injections > 0:
            efficiency_factor = 1.0 / math.sqrt(self.attempt_count)
            reward += self.efficiency_bonus * efficiency_factor
        
        return reward
    
    def _calculate_exploration_rewards(self, action: int, analysis: Dict[str, Any]) -> float:
        """
        Calculate exploration rewards for trying new actions.
        
        Args:
            action: Action taken
            analysis: Response analysis results
            
        Returns:
            Exploration reward
        """
        reward = 0.0
        
        # Action diversity reward
        self.action_frequency[action] = self.action_frequency.get(action, 0) + 1
        
        # Reward for trying less frequent actions
        total_actions = sum(self.action_frequency.values())
        action_prob = self.action_frequency[action] / total_actions
        
        # Inverse frequency bonus (more reward for rare actions)
        if action_prob < 0.1:  # Action used less than 10% of the time
            reward += self.exploration_reward * (0.1 - action_prob) / 0.1
        
        # Novel response reward
        if analysis.get('response_anomaly', False):
            reward += self.exploration_reward
        
        return reward
    
    def _calculate_penalties(self, analysis: Dict[str, Any], action: int, response: Dict[str, Any]) -> float:
        """
        Calculate penalties for negative outcomes.
        
        Args:
            analysis: Response analysis results
            action: Action taken
            response: HTTP response
            
        Returns:
            Penalty value (negative)
        """
        penalty = 0.0
        
        # Failed attempt penalty
        if not analysis.get('injection_detected', False) and not analysis.get('error_detected', False):
            penalty += self.failed_attempt_penalty
        
        # Repetitive action penalty
        if self.action_frequency.get(action, 0) > 5:
            penalty += self.repetitive_action_penalty * (self.action_frequency[action] - 5)
        
        # Time-based penalty for slow responses
        response_time = response.get('response_time', 0)
        if response_time > 5.0:
            penalty += self.time_penalty * response_time
        
        # HTTP error penalty
        status_code = response.get('status_code', 200)
        if status_code >= 500:
            penalty += -5.0  # Server error penalty
        elif status_code == 404:
            penalty += -2.0  # Not found penalty
        
        return penalty
    
    def _calculate_bonus_rewards(self, state: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """
        Calculate bonus rewards for exceptional performance.
        
        Args:
            state: Current state
            analysis: Response analysis results
            
        Returns:
            Bonus reward
        """
        reward = 0.0
        
        # Stealth bonus (success without triggering WAF)
        if (analysis.get('injection_detected', False) and 
            not analysis.get('waf_triggered', False) and 
            state.get('waf_detected', False)):
            reward += 25.0
        
        # Multi-vector success bonus
        success_types = sum([
            analysis.get('injection_detected', False),
            analysis.get('error_detected', False),
            analysis.get('blind_injection_possible', False)
        ])
        if success_types >= 2:
            reward += 15.0 * success_types
        
        # Consistency bonus (multiple successes in a row)
        if hasattr(self, 'consecutive_successes'):
            if analysis.get('injection_detected', False):
                self.consecutive_successes += 1
                if self.consecutive_successes >= 3:
                    reward += 10.0 * self.consecutive_successes
            else:
                self.consecutive_successes = 0
        else:
            self.consecutive_successes = 1 if analysis.get('injection_detected', False) else 0
        
        return reward
    
    def _update_tracking(self, action: int, analysis: Dict[str, Any]):
        """
        Update internal tracking variables.
        
        Args:
            action: Action taken
            analysis: Response analysis results
        """
        # Update action frequency
        self.action_frequency[action] = self.action_frequency.get(action, 0) + 1
        
        # Track successful injection types
        if analysis.get('injection_detected', False):
            injection_type = self._determine_injection_type(analysis)
            if not hasattr(self, 'injection_types'):
                self.injection_types = set()
            self.injection_types.add(injection_type)
    
    def _determine_injection_type(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the type of injection based on analysis.
        
        Args:
            analysis: Response analysis results
            
        Returns:
            Injection type string
        """
        if analysis.get('error_detected', False):
            return 'error_based'
        elif analysis.get('blind_injection_possible', False):
            return 'blind'
        else:
            return 'union_based'
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """
        Get statistics about rewards given.
        
        Returns:
            Dictionary with reward statistics
        """
        return {
            'total_attempts': self.attempt_count,
            'successful_injections': self.successful_injections,
            'success_rate': self.successful_injections / max(self.attempt_count, 1),
            'action_frequency': self.action_frequency.copy(),
            'discoveries_made': len(self.previous_discoveries),
            'discovered_items': list(self.previous_discoveries)
        }
    
    def reset(self):
        """
        Reset the reward calculator for a new episode.
        """
        self.previous_discoveries.clear()
        self.action_frequency.clear()
        self.attempt_count = 0
        self.successful_injections = 0
        if hasattr(self, 'consecutive_successes'):
            self.consecutive_successes = 0
        if hasattr(self, 'recent_waf_triggers'):
            self.recent_waf_triggers = []
