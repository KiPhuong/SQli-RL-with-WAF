"""
Main Training Loop for SQL Injection RL Agent
Orchestrates the training process with DQN and Boltzmann exploration
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
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
            'temperature_decay': 0.995,
            'save_frequency': 100,
            'log_frequency': 10,
            'model_save_path': 'models/',
            'log_save_path': 'logs/'
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Create directories
        os.makedirs(self.config['model_save_path'], exist_ok=True)
        os.makedirs(self.config['log_save_path'], exist_ok=True)
        
        # Initialize environment
        self.env = SQLiEnvironment(
            target_url=self.config['target_url'],
            parameter=self.config['parameter'],
            method=self.config['method'],
            max_steps=self.config['max_steps_per_episode']
        )
        
        # Initialize agent
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()
        
        agent_config = {
            'learning_rate': self.config['learning_rate'],
            'gamma': self.config['gamma'],
            'memory_size': self.config['memory_size'],
            'batch_size': self.config['batch_size'],
            'target_update_freq': self.config['target_update_freq'],
            'initial_temperature': self.config['initial_temperature'],
            'min_temperature': self.config['min_temperature'],
            'temperature_decay': self.config['temperature_decay']
        }
        
        self.agent = SQLiRLAgent(state_size, action_size, agent_config)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.exploration_temps = []
        
        print(f"Initialized SQLi RL Trainer")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Target URL: {self.config['target_url']}")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['num_episodes']} episodes...")
        
        best_reward = float('-inf')
        
        for episode in range(self.config['num_episodes']):
            episode_reward, episode_length, episode_success = self._run_episode(episode)
            
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
    
    def _run_episode(self, episode_num: int) -> tuple:
        """Run a single episode"""
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        episode_success = False
        
        while True:
            # Agent selects action
            action = self.agent.select_token(state)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.memory) > self.agent.config['batch_size']:
                self.agent.replay()
            
            # Update for next iteration
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Check for success
            if info.get('sqli_detected', False):
                episode_success = True
            
            if done:
                break
        
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


def main():
    """Main function"""
    # Configuration
    config = {
        'target_url': 'https://www.zixem.altervista.org/SQLi/level1.php?id=1',  # Change this to your target
        'parameter': 'id',
        'method': 'GET',
        'num_episodes': 1000,
        'max_steps_per_episode': 50,
        'learning_rate': 0.001,
        'initial_temperature': 2.0,
        'save_frequency': 100,
        'log_frequency': 10
    }
    
    # Create trainer
    trainer = SQLiRLTrainer(config)
    
    # Train agent
    trainer.train()
    
    # Test agent (optional)
    # trainer.test_agent('models/final_model.pth', num_episodes=5)


if __name__ == "__main__":
    main()
