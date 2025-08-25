"""
Main Training Loop for SQL Injection RL Agent
Orchestrates the training process with DQN and Boltzmann exploration
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

from agent import SQLiRLAgent
from env import SQLiEnvironment
from gen_action import GenAction
from bypass_waf import BypassWAF


class SQLiRLTrainer:
    """Main trainer class for SQL injection RL agent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default configuration
        default_config = {
            'target_url': 'http://localhost:8080/vuln',
            'parameter': 'id',
            'injection_point': '1',  # Base value to inject after
            'method': 'GET',
            'max_steps_per_episode': 50,
            'num_episodes': 1000,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'initial_temperature': 2.0,
            'min_temperature': 0.1,
            'temperature_decay': 0.9999,
            'save_frequency': 100,
            'log_frequency': 10,
            'model_save_path': 'models/',
            'log_save_path': 'logs/',
            'debug_mode': True,  # Enable debug output
            'debug_frequency': 1,  # Debug every N steps
            'resume_from_model': None,  # hoặc None nếu không muốn load
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Create directories
        os.makedirs(self.config['model_save_path'], exist_ok=True)
        os.makedirs(self.config['log_save_path'], exist_ok=True)

        # Setup debug logger if debug mode is enabled
        self.debug_logger = None
        if self.config.get('debug_mode', False):
            self._setup_debug_logger()

        # Initialize environment
        self.env = SQLiEnvironment(
            target_url=self.config['target_url'],
            parameter=self.config['parameter'],
            method=self.config['method'],
            injection_point=self.config.get('injection_point', '1'),
            max_steps=self.config['max_steps_per_episode'],
            blocked_keywords=self.config.get('blocked_keywords', None)
        )
        
        # Initialize agent with dynamic sizing
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()

        # Adjust training parameters based on action space size
        adjusted_config = self._adjust_config_for_action_space(action_size)

        agent_config = {
            'learning_rate': adjusted_config['learning_rate'],
            'gamma': self.config['gamma'],
            'memory_size': adjusted_config['memory_size'],
            'batch_size': adjusted_config['batch_size'],
            'target_update_freq': self.config['target_update_freq'],
            'initial_temperature': adjusted_config['initial_temperature'],
            'min_temperature': self.config['min_temperature'],
            'temperature_decay': adjusted_config['temperature_decay']
        }
        
        self.agent = SQLiRLAgent(state_size, action_size, agent_config)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.exploration_temps = []
        self.payloads = []
        #self.tokens = []
        
        print(f"🚀 Initialized SQLi RL Trainer")
        print(f"📊 State size: {state_size}")
        print(f"🎯 Action size: {action_size}")
        print(f"🌐 Target URL: {self.config['target_url']}")
        print(f"🔧 Debug mode: {'✅ ON' if self.config['debug_mode'] else '❌ OFF'}")

        if self.config['debug_mode']:
            print(f"\n🔍 DEBUG CONFIGURATION:")
            print(f"  • Debug frequency: Every {self.config['debug_frequency']} step(s)")
            print(f"  • Max steps per episode: {self.config['max_steps_per_episode']}")
            print(f"  • Log frequency: Every {self.config['log_frequency']} episode(s)")

            # Show some sample tokens
            sample_tokens = self.env.gen_action.action_space.get_all_tokens()[:10]
            print(f"  • Sample tokens: {sample_tokens}")
            print(f"  • Total vocabulary size: {len(self.env.gen_action.action_space.get_all_tokens())}")
            

            # Show baseline info
            if self.env.baseline_response:
                print(f"  • Baseline status: {self.env.baseline_response['status_code']}")
                print(f"  • Baseline length: {self.env.baseline_response['content_length']} chars")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['num_episodes']} episodes...")
        # Thêm đoạn này để load model nếu có
        resume_path = self.config.get('retrain_model', None)
        if resume_path:
            print(f"🔄 Loading model weights from: {resume_path}")
            self.agent.load_model(resume_path)

        
        best_reward = float('-inf')
        
        for episode in range(self.config['num_episodes']):
            episode_reward, episode_length, episode_success = self._run_episode(episode)

            # Decay temperature after each episode
            self.agent.decay_temperature()

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_episodes.append(episode_success)
            self.exploration_temps.append(self.agent.exploration.temperature)

            
            # Log progress
            if episode % self.config['log_frequency'] == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                success_rate = np.mean(self.success_episodes[-100:])
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:6.2f} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Success Rate: {success_rate:5.2%} | "
                      f"Temperature: {self.agent.exploration.temperature:.3f} | "
                      f"Steps: {episode_length}")
            
            # Save model periodically
            if episode % self.config['save_frequency'] == 0 and episode > 0:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    model_path = os.path.join(self.config['model_save_path'], 
                                            f'best_model_episode_{episode}.pth')
                    self.agent.save_model(model_path)
                    print(f"Saved best model with reward {episode_reward:.2f}")
        
        # Save final model
        final_model_path = os.path.join(self.config['model_save_path'], 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        # Save training logs
        self._save_training_logs()
        
        # Plot results
        self._plot_training_results()
        
        print("\nTraining completed!")

        # Close debug logger
        if self.debug_logger:
            self.debug_logger.info("")
            self.debug_logger.info("🎯 TRAINING COMPLETED!")
            self.debug_logger.info("=" * 80)
            # Close all handlers
            for handler in self.debug_logger.handlers[:]:
                handler.close()
                self.debug_logger.removeHandler(handler)

    def _setup_debug_logger(self):
        """Setup debug logger to write to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_file = os.path.join(self.config['log_save_path'], f'debug_{timestamp}.log')

        # Create logger
        self.debug_logger = logging.getLogger('sqli_debug')
        self.debug_logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in self.debug_logger.handlers[:]:
            self.debug_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(debug_log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.debug_logger.addHandler(file_handler)

        print(f"📝 Debug logging enabled: {debug_log_file}")
        self.debug_logger.info("=" * 80)
        self.debug_logger.info("🚀 SQL INJECTION RL TRAINING DEBUG LOG")
        self.debug_logger.info("=" * 80)
        self.debug_logger.info(f"Target URL: {self.config['target_url']}")
        self.debug_logger.info(f"Parameter: {self.config['parameter']}")
        self.debug_logger.info(f"Episodes: {self.config['num_episodes']}")
        self.debug_logger.info(f"Max steps per episode: {self.config['max_steps_per_episode']}")
        self.debug_logger.info("=" * 80)


    def _run_episode(self, episode_num: int) -> tuple:
        """Run a single episode"""
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        episode_success = False

        if self.config['debug_mode'] and self.debug_logger:
            self.debug_logger.info("")
            self.debug_logger.info("=" * 60)
            self.debug_logger.info(f"🚀 EPISODE {episode_num} STARTED")
            self.debug_logger.info("=" * 60)
            self.debug_logger.info(f"Initial state shape: {np.array(state).shape}")
            self.debug_logger.info(f"Initial state (first 10 tokens): {state[:10]}")
        step_idx = 0  
        prev_token = None
        while True:
            step_count += 1
            # Agent selects action

            action = self.agent.select_token(state, step_idx, prev_token)
            prev_token = self.agent.id_to_token.get(action, 'UNK')
            step_idx +=1
            # Get Q-values for debugging
            if self.config['debug_mode'] and self.debug_logger and step_count % self.config['debug_frequency'] == 0:
                q_values = self.agent.get_q_values(state)
                top_5_actions = np.argsort(q_values)[-5:][::-1]

                self.debug_logger.info("")
                self.debug_logger.info(f"🔍 STEP {step_count} DEBUG INFO:")
                self.debug_logger.info("─" * 50)

                # Show current state info
                current_payload = self.env.current_payload

                self.debug_logger.info("📊 Current State:")
                self.debug_logger.info(f"  • State vector size: {len(state)}")
                self.debug_logger.info(f"  • Current payload: '{current_payload}'")
                self.debug_logger.info(f"  • Payload length: {len(current_payload)} chars")
                self.debug_logger.info(f"  • State range: [{state.min():.3f}, {state.max():.3f}]")

                # Show Q-values and action selection
                self.debug_logger.info("🧠 Agent Decision:")
                self.debug_logger.info(f"  • Selected action (token ID): {action}")
                self.debug_logger.info(f"  • Selected token: '{self.env.gen_action.get_token_name(action)}'")
                self.debug_logger.info(f"  • Q-value for selected action: {q_values[action]:.4f}")
                self.debug_logger.info(f"  • Temperature: {self.agent.exploration.temperature:.4f}")

                self.debug_logger.info("  • Top 5 Q-values:")
                for i, act_id in enumerate(top_5_actions):
                    token_name = self.env.gen_action.get_token_name(act_id)
                    self.debug_logger.info(f"    {i+1}. Token '{token_name}' (ID:{act_id}) = {q_values[act_id]:.4f}")

            # Environment step
            next_state, reward, done, info = self.env.step(action)

            #print(f"Current Payload: {self.env.current_payload}")
            self.payloads.append(self.env.current_payload)

            # Debug bypass processing results
            if self.config['debug_mode'] and self.debug_logger and step_count % self.config['debug_frequency'] == 0:
                self.debug_logger.info("🔧 Bypass Processing:")
                self.debug_logger.info(f"  • Original token: '{info.get('original_token', 'N/A')}'")
                self.debug_logger.info(f"  • Processed token: '{info.get('processed_token', 'N/A')}'")
                self.debug_logger.info(f"  • Bypass applied: {info.get('bypass_applied', False)}")
                if info.get('bypass_method'):
                    self.debug_logger.info(f"  • Bypass method: {info['bypass_method']}")
                    self.debug_logger.info(f"  • Bypass success: {info.get('bypass_success', 'N/A')}")

            # Debug environment step results
            if self.config['debug_mode'] and self.debug_logger and step_count % self.config['debug_frequency'] == 0:
                self.debug_logger.info("🌐 Environment Response:")
                self.debug_logger.info(f"  • Final URL: '{info.get('final_url', 'N/A')}'")
                self.debug_logger.info(f"  • Final payload: '{info['payload']}'")
                self.debug_logger.info(f"  • HTTP status: {info['response_status']}")
                self.debug_logger.info(f"  • Response length: {info['response_length']} chars")
                self.debug_logger.info(f"  • Response time: {info['response_time']:.3f}s")
                self.debug_logger.info(f"  • WAF blocked: {info['is_blocked']}")
                self.debug_logger.info(f"  • SQL error detected: {info['error_detected']}")
                self.debug_logger.info(f"  • SQLi success detected: {info['sqli_detected']}")
                self.debug_logger.info(f"  • Reward: {reward:.2f}")

                # Enhanced error information display
                if info['error_detected']:
                    self.debug_logger.info("🔍 SQL Error Analysis:")
                    self.debug_logger.info(f"  • Database type: {info.get('database_type', 'unknown')}")

                    if info.get('extracted_columns'):
                        self.debug_logger.info(f"  • Extracted columns: {info['extracted_columns']}")

                    if info.get('extracted_tables'):
                        self.debug_logger.info(f"  • Extracted tables: {info['extracted_tables']}")

                    if info.get('error_messages'):
                        self.debug_logger.info(f"  • Error messages: {info['error_messages'][:2]}")  # Show first 2 messages

                    error_info = info.get('error_info', {})
                    if error_info.get('detected_patterns'):
                        self.debug_logger.info(f"  • Detected patterns: {error_info['detected_patterns'][:3]}")  # Show first 3 patterns

                # Show response content preview if there's an error or success
                if info['error_detected'] or info['sqli_detected'] or info['is_blocked']:
                    self.debug_logger.info(f"  • Response preview: '{info.get('response_content_preview', 'N/A')}'")

                # Show baseline comparison
                if info.get('baseline_status') and info.get('baseline_length'):
                    status_diff = info['response_status'] != info['baseline_status']
                    length_diff = abs(info['response_length'] - info['baseline_length'])
                    self.debug_logger.info("  • Baseline comparison:")
                    self.debug_logger.info(f"    - Status changed: {'✅' if status_diff else '❌'} ({info['baseline_status']} → {info['response_status']})")
                    self.debug_logger.info(f"    - Length diff: {length_diff} chars ({info['baseline_length']} → {info['response_length']})")

                # Show simple state information
                self.debug_logger.info("📈 State Update:")
                self.debug_logger.info(f"  • State vector length: {len(next_state)}")
                self.debug_logger.info(f"  • State features (first 10): {next_state[:10]}")
                self.debug_logger.info(f"  • Updated payload: '{info['payload']}'")
                self.debug_logger.info(f"  • Payload length: {len(info['payload'])} chars")
                self.debug_logger.info(f"  • Episode done: {done}")

                # Show state breakdown if debug
                if hasattr(self.env, 'state_manager'):
                    debug_info = self.env.state_manager.debug_state(next_state)
                    self.debug_logger.info(f"  • Payload features: {list(debug_info['payload_features'].values())[:5]}...")
                    self.debug_logger.info(f"  • Response features: {list(debug_info['response_features'].values())[:5]}...")
                    self.debug_logger.info(f"  • WAF features: {list(debug_info['waf_features'].values())[:5]}...") 

                if done:
                    self.debug_logger.info("🏁 Episode termination reason:")
                    if info['sqli_detected']:
                        self.debug_logger.info("  ✅ SQL injection success!")
                    elif step_count >= self.config['max_steps_per_episode']:
                        self.debug_logger.info("  ⏰ Maximum steps reached")
                    elif info.get('is_complete', False):
                        self.debug_logger.info("  🔚 END_TOKEN reached")
                    elif info.get('is_full', False):
                        self.debug_logger.info("  📦 State is full")

            # Store experience
            self.agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(self.agent.memory) > self.agent.config['batch_size']:
                self.agent.replay()

            # Update for next iteration
            state = next_state
            total_reward += reward

            # Check for success
            if info.get('sqli_detected', False):
                episode_success = True

            if done:
                break

        if self.config['debug_mode'] and self.debug_logger:
            self.debug_logger.info("")
            self.debug_logger.info(f"🏆 EPISODE {episode_num} SUMMARY:")
            self.debug_logger.info(f"  • Total steps: {step_count}")
            self.debug_logger.info(f"  • Total reward: {total_reward:.2f}")
            self.debug_logger.info(f"  • Success: {'✅ YES' if episode_success else '❌ NO'}")
            self.debug_logger.info(f"  • Final payload: '{info.get('payload', '')}'")
            self.debug_logger.info("=" * 60)

        return total_reward, step_count, episode_success
    
    def _save_training_logs(self):
        """Save training metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config['log_save_path'], f'training_log_{timestamp}.json')
        
        log_data = {
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_episodes': self.success_episodes,
            'exploration_temps': self.exploration_temps,
            'payloads': self.payloads,
            'final_stats': {
                'total_episodes': len(self.episode_rewards),
                'average_reward': np.mean(self.episode_rewards),
                'success_rate': np.mean(self.success_episodes),
                'final_temperature': self.exploration_temps[-1] if self.exploration_temps else 0
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training logs saved to {log_file}")

    def _plot_training_results(self):
        """Plot training results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        ax1.plot(self.episode_rewards, alpha=0.6)
        ax1.plot(self._moving_average(self.episode_rewards, 100), 'r-', linewidth=2)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)

        # Success rate
        success_ma = self._moving_average([float(x) for x in self.success_episodes], 100)
        ax2.plot(success_ma, 'g-', linewidth=2)
        ax2.set_title('Success Rate (100-episode moving average)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.grid(True)

        # Episode lengths
        ax3.plot(self.episode_lengths, alpha=0.6)
        ax3.plot(self._moving_average(self.episode_lengths, 100), 'b-', linewidth=2)
        ax3.set_title('Episode Lengths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True)

        # Exploration temperature
        ax4.plot(self.exploration_temps, 'orange', linewidth=2)
        ax4.set_title('Exploration Temperature')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Temperature')
        ax4.grid(True)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.config['log_save_path'], f'training_results_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Training plots saved to {plot_file}")

    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data

        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(data[start_idx:i+1]))

        return moving_avg

    def test_agent(self, model_path: str, num_episodes: int = 10):
        """Test trained agent"""
        print(f"\nTesting agent for {num_episodes} episodes...")
        
        # Load model
        self.agent.load_model(model_path)
        
        # Disable exploration for testing
        original_temp = self.agent.exploration.temperature
        self.agent.exploration.temperature = self.agent.exploration.min_temperature
        
        test_rewards = []
        test_successes = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_success = False
            
            while True:
                action = self.agent.select_token(state)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                
                if info.get('sqli_detected', False):
                    episode_success = True
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            test_successes.append(episode_success)
            
            print(f"Test Episode {episode+1}: Reward={total_reward:.2f}, "
                  f"Success={'Yes' if episode_success else 'No'}")
        
        # Restore original temperature
        self.agent.exploration.temperature = original_temp
        
        print(f"\nTest Results:")
        print(f"Average Reward: {np.mean(test_rewards):.2f}")
        print(f"Success Rate: {np.mean(test_successes):.2%}")

    def _adjust_config_for_action_space(self, action_size: int) -> Dict[str, Any]:
        """Adjust training configuration based on action space size"""
        base_config = {
            'learning_rate': self.config['learning_rate'],
            'memory_size': self.config['memory_size'],
            'batch_size': self.config['batch_size'],
            'initial_temperature': self.config['initial_temperature'],
            'temperature_decay': self.config['temperature_decay']
        }

        print(f"🔧 Adjusting config for action space size: {action_size}")

        if action_size <= 100:
            # Small action space - default settings
            print("   → Using default settings (small action space)")
            return base_config
        elif action_size <= 500:
            # Medium action space - need more exploration and memory
            adjusted = {
                **base_config,
                'learning_rate': base_config['learning_rate'] * 0.8,
                'memory_size': base_config['memory_size'] * 2,
                'batch_size': min(base_config['batch_size'] * 2, 128),
                'initial_temperature': base_config['initial_temperature'] * 1.5,
                'temperature_decay': max(base_config['temperature_decay'] * 0.9999, 0.98)
            }
            print("   → Adjusted for medium action space")
            return adjusted
        else:
            # Large action space - significant adjustments
            adjusted = {
                **base_config,
                'learning_rate': base_config['learning_rate'] * 0.5,
                'memory_size': base_config['memory_size'] * 5,
                'batch_size': min(base_config['batch_size'] * 4, 256),
                'initial_temperature': base_config['initial_temperature'] * 2.0,
                'temperature_decay': max(base_config['temperature_decay'] * 0.9999, 0.98)
            }
            print("   → Adjusted for large action space")
            return adjusted


def main():
    """Main function"""
    # Configuration
    config = {
        'target_url': 'https://www.zixem.altervista.org/SQLi/level1.php',  # Base URL without parameters
        'parameter': 'id',
        'injection_point': '1',  # Will inject after id=1
        'method': 'GET',
        'num_episodes': 10,  # Reduced for debugging
        'max_steps_per_episode': 20,  # Reduced for debugging
        'learning_rate': 0.001,
        'initial_temperature': 2.0,
        'save_frequency': 5,  # Save more frequently for debugging
        'log_frequency': 1,   # Log every episode for debugging
        'debug_mode': True,   # Enable debug output
        'debug_frequency': 1  # Debug every step
    }
    
    # Create trainer
    trainer = SQLiRLTrainer(config)
    
    # Train agent
    trainer.train()
    
    # Test agent (optional)
    # trainer.test_agent('models/final_model.pth', num_episodes=5)


if __name__ == "__main__":
    main()
