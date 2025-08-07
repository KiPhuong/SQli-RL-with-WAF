"""
Main application file for SQL Injection RL with WAF bypass framework.
This file orchestrates the entire training and testing pipeline.
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, Optional
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.rl_agent import RLAgent
from src.agent.action_space import ActionSpace
from src.environment.pentest_env import PentestEnvironment
from src.environment.state_manager import StateManager
from src.environment.reward_calculator import RewardCalculator
from src.waf.waf_detector import WAFDetector
from src.waf.bypass_methods import BypassMethods
from src.payloads.payload_generator import PayloadGenerator
from src.payloads.payload_catalog import PayloadCatalog
from src.guidelines.catalog_parser import CatalogParser
from src.guidelines.rule_engine import RuleEngine
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.network_utils import NetworkUtils


class SQLiRLFramework:
    """
    Main framework class that orchestrates the SQL injection RL system.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the framework.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logging(
            log_dir=self.config.get('logging', {}).get('log_dir', 'logs'),
            log_level=self.config.get('logging', {}).get('level', 'INFO')
        )
        
        # Initialize components
        self.action_space = ActionSpace()
        self.network_utils = NetworkUtils(
            timeout=self.config.get('network', {}).get('timeout', 10),
            max_retries=self.config.get('network', {}).get('max_retries', 3)
        )
        
        # Initialize WAF and payload components
        self.waf_detector = WAFDetector()
        self.bypass_methods = BypassMethods()
        self.payload_generator = PayloadGenerator(
            self.action_space, 
            self.bypass_methods
        )
        self.payload_catalog = PayloadCatalog()
        
        # Initialize environment components
        self.state_manager = StateManager()
        self.reward_calculator = RewardCalculator()
        
        # Initialize guidelines and rules
        self.catalog_parser = CatalogParser()
        self.rule_engine = RuleEngine()
        
        # Initialize environment and agent
        env_config = self.config.get('environment', {})
        self.environment = PentestEnvironment(
            target_url=env_config.get('target_url', 'http://testphp.vulnweb.com'),
            state_manager=self.state_manager,
            reward_calculator=self.reward_calculator,
            waf_detector=self.waf_detector,
            payload_generator=self.payload_generator,
            network_utils=self.network_utils
        )
        
        agent_config = self.config.get('agent', {})
        self.agent = RLAgent(
            state_size=self.state_manager.get_state_size(),
            action_size=self.action_space.get_action_count(),
            learning_rate=agent_config.get('learning_rate', 0.001),
            memory_size=agent_config.get('memory_size', 10000),
            batch_size=agent_config.get('batch_size', 32)
        )
        
        self.logger.main_logger.info("Framework initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            # Return default configuration
            return {
                'agent': {
                    'learning_rate': 0.001,
                    'memory_size': 10000,
                    'batch_size': 32,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.1,
                    'epsilon_decay': 0.995,
                    'target_update_frequency': 100
                },
                'environment': {
                    'target_url': 'http://testphp.vulnweb.com',
                    'max_steps_per_episode': 100,
                    'timeout': 10
                },
                'training': {
                    'episodes': 1000,
                    'save_frequency': 100,
                    'eval_frequency': 50,
                    'early_stopping_patience': 200
                },
                'logging': {
                    'log_dir': 'logs',
                    'level': 'INFO'
                },
                'network': {
                    'timeout': 10,
                    'max_retries': 3
                }
            }
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def train(self, episodes: int = None, save_model_path: str = None):
        """
        Train the RL agent.
        
        Args:
            episodes: Number of episodes to train (overrides config)
            save_model_path: Path to save trained model
        """
        training_config = self.config.get('training', {})
        episodes = episodes or training_config.get('episodes', 1000)
        save_frequency = training_config.get('save_frequency', 100)
        eval_frequency = training_config.get('eval_frequency', 50)
        
        self.logger.main_logger.info(f"Starting training for {episodes} episodes")
        
        # Training statistics
        episode_rewards = []
        episode_steps = []
        success_count = 0
        best_reward = float('-inf')
        
        for episode in range(episodes):
            # Reset environment
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Agent selects action
                action = self.agent.act(state)
                
                # Take action in environment
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state and tracking
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train agent if enough experiences
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()
            
            # Episode completed
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            # Check for success
            if info.get('injection_found', False):
                success_count += 1
            
            # Log episode completion
            exploration_rate = getattr(self.agent, 'epsilon', 0.0)
            self.logger.log_training_episode(
                episode + 1, total_reward, steps, 
                info.get('injection_found', False), exploration_rate
            )
            
            # Update target network
            if episode % self.agent.target_update_frequency == 0:
                self.agent.update_target_network()
            
            # Evaluate and save model
            if (episode + 1) % eval_frequency == 0:
                avg_reward = sum(episode_rewards[-eval_frequency:]) / eval_frequency
                success_rate = success_count / (episode + 1)
                
                self.logger.main_logger.info(
                    f"Episode {episode + 1}: Avg Reward={avg_reward:.2f}, "
                    f"Success Rate={success_rate:.2%}"
                )
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    if save_model_path:
                        self.agent.save_model(save_model_path.replace('.pth', '_best.pth'))
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0 and save_model_path:
                self.agent.save_model(save_model_path.replace('.pth', f'_episode_{episode + 1}.pth'))
        
        # Save final model
        if save_model_path:
            self.agent.save_model(save_model_path)
        
        # Training summary
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        success_rate = success_count / episodes
        
        self.logger.main_logger.info(
            f"Training completed! Average Reward: {avg_reward:.2f}, "
            f"Success Rate: {success_rate:.2%}"
        )
        
        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps
        }
    
    def test(self, model_path: str = None, episodes: int = 10):
        """
        Test the trained agent.
        
        Args:
            model_path: Path to trained model
            episodes: Number of test episodes
            
        Returns:
            Test results
        """
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
            self.logger.main_logger.info(f"Loaded model from {model_path}")
        else:
            self.logger.main_logger.warning("No model path provided or file not found")
        
        # Disable exploration for testing
        original_epsilon = getattr(self.agent, 'epsilon', 0.0)
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = 0.0
        
        self.logger.main_logger.info(f"Starting testing for {episodes} episodes")
        
        test_results = {
            'episodes': episodes,
            'total_rewards': [],
            'steps_taken': [],
            'injections_found': 0,
            'waf_bypasses': 0,
            'successful_episodes': 0
        }
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.environment.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Record results
            test_results['total_rewards'].append(total_reward)
            test_results['steps_taken'].append(steps)
            
            if info.get('injection_found', False):
                test_results['injections_found'] += 1
                test_results['successful_episodes'] += 1
            
            if info.get('waf_bypassed', False):
                test_results['waf_bypasses'] += 1
            
            self.logger.main_logger.info(
                f"Test Episode {episode + 1}: Reward={total_reward:.2f}, "
                f"Steps={steps}, Success={info.get('injection_found', False)}"
            )
        
        # Calculate statistics
        test_results['avg_reward'] = sum(test_results['total_rewards']) / episodes
        test_results['avg_steps'] = sum(test_results['steps_taken']) / episodes
        test_results['success_rate'] = test_results['successful_episodes'] / episodes
        test_results['injection_rate'] = test_results['injections_found'] / episodes
        test_results['bypass_rate'] = test_results['waf_bypasses'] / episodes
        
        # Restore original epsilon
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = original_epsilon
        
        self.logger.main_logger.info(
            f"Testing completed! Success Rate: {test_results['success_rate']:.2%}, "
            f"Avg Reward: {test_results['avg_reward']:.2f}"
        )
        
        return test_results
    
    def interactive_mode(self, model_path: str = None):
        """
        Run framework in interactive mode for manual testing.
        
        Args:
            model_path: Path to trained model
        """
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
            self.logger.main_logger.info(f"Loaded model from {model_path}")
        
        print("=== SQL Injection RL Framework - Interactive Mode ===")
        print("Commands:")
        print("  test <url> - Test a specific URL")
        print("  payload <payload> - Test a specific payload")
        print("  waf <url> - Detect WAF on URL")
        print("  stats - Show framework statistics")
        print("  quit - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input("SQli-RL> ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                
                elif command[0] == 'test' and len(command) > 1:
                    url = command[1]
                    self._interactive_test_url(url)
                
                elif command[0] == 'payload' and len(command) > 1:
                    payload = ' '.join(command[1:])
                    self._interactive_test_payload(payload)
                
                elif command[0] == 'waf' and len(command) > 1:
                    url = command[1]
                    self._interactive_detect_waf(url)
                
                elif command[0] == 'stats':
                    self._interactive_show_stats()
                
                else:
                    print("Unknown command or missing arguments")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _interactive_test_url(self, url: str):
        """Test a URL in interactive mode."""
        print(f"Testing URL: {url}")
        
        # Update environment target
        self.environment.target_url = url
        
        # Run a single episode
        state = self.environment.reset()
        actions_taken = []
        done = False
        step = 0
        
        while not done and step < 20:  # Limit steps in interactive mode
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            
            action_name = self.action_space.get_action_name(action)
            actions_taken.append((action, action_name, reward))
            
            print(f"  Step {step + 1}: {action_name} (reward: {reward:.2f})")
            
            state = next_state
            step += 1
        
        # Show results
        if info.get('injection_found'):
            print(f"✓ SQL Injection found! Type: {info.get('injection_type', 'Unknown')}")
        else:
            print("✗ No SQL injection found")
        
        if info.get('waf_detected'):
            print(f"⚠ WAF detected: {info.get('waf_type', 'Unknown')}")
    
    def _interactive_test_payload(self, payload: str):
        """Test a specific payload in interactive mode."""
        print(f"Testing payload: {payload}")
        
        # Send direct request
        response = self.network_utils.send_request(
            self.environment.target_url,
            method='GET',
            params={'id': payload}
        )
        
        # Analyze response
        analysis = self.state_manager.analyze_response(response)
        
        print(f"Status Code: {response.get('status_code')}")
        print(f"Response Time: {response.get('response_time', 0):.2f}s")
        print(f"Content Length: {response.get('content_length', 0)} bytes")
        
        if analysis.get('sql_error_detected'):
            print("✓ SQL error detected in response")
        
        if analysis.get('waf_blocked'):
            print("⚠ Request appears to be blocked by WAF")
    
    def _interactive_detect_waf(self, url: str):
        """Detect WAF on a URL in interactive mode."""
        print(f"Detecting WAF on: {url}")
        
        waf_info = self.waf_detector.detect_waf(url)
        
        if waf_info['detected']:
            print(f"✓ WAF detected: {waf_info['type']}")
            print(f"  Confidence: {waf_info['confidence']:.2f}")
            print(f"  Detection method: {waf_info['detection_method']}")
        else:
            print("✗ No WAF detected")
    
    def _interactive_show_stats(self):
        """Show framework statistics in interactive mode."""
        print("=== Framework Statistics ===")
        
        # Agent statistics
        if hasattr(self.agent, 'memory'):
            print(f"Agent memory size: {len(self.agent.memory)}")
        
        # Payload statistics
        catalog_stats = self.payload_catalog.get_statistics()
        print(f"Payload catalog: {catalog_stats.get('total_payloads', 0)} payloads")
        
        # Network statistics
        network_stats = self.network_utils.get_request_statistics()
        print(f"Requests sent: {network_stats.get('total_requests', 0)}")
        
        # Log statistics
        log_stats = self.logger.get_log_statistics()
        print(f"Log files: {len(log_stats.get('log_files', {}))}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='SQL Injection RL with WAF bypass framework')
    parser.add_argument('mode', choices=['train', 'test', 'interactive'], 
                       help='Operation mode')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--model', help='Model file path')
    parser.add_argument('--episodes', type=int, help='Number of episodes')
    parser.add_argument('--target', help='Target URL for testing')
    parser.add_argument('--save-model', help='Path to save trained model')
    
    args = parser.parse_args()
    
    try:
        # Initialize framework
        framework = SQLiRLFramework(args.config)
        
        if args.target:
            framework.environment.target_url = args.target
        
        if args.mode == 'train':
            framework.train(
                episodes=args.episodes,
                save_model_path=args.save_model or 'models/trained_model.pth'
            )
        
        elif args.mode == 'test':
            if not args.model:
                print("Error: Model path required for testing mode")
                return 1
            
            results = framework.test(
                model_path=args.model,
                episodes=args.episodes or 10
            )
            
            print("\n=== Test Results ===")
            print(f"Success Rate: {results['success_rate']:.2%}")
            print(f"Average Reward: {results['avg_reward']:.2f}")
            print(f"Injections Found: {results['injections_found']}")
            print(f"WAF Bypasses: {results['waf_bypasses']}")
        
        elif args.mode == 'interactive':
            framework.interactive_mode(args.model)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
