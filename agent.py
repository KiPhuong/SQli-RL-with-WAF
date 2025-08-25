import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any
from env import SQLiEnvironment
from gen_action import ActionSpace
import re

class DQN(nn.Module):
    """Deep Q-Network for token selection"""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(DQN, self).__init__()
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


class BoltzmannExploration:
    """Boltzmann (Softmax) exploration strategy"""
    def __init__(self, initial_temperature: float = 2.0, min_temperature: float = 0.1, decay_rate: float = 0.99):
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate

    def select_action(self, q_values: np.ndarray) -> int:
        scaled_q_values = q_values / max(self.temperature, self.min_temperature)
        exp_values = np.exp(scaled_q_values - np.max(scaled_q_values))
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probabilities)

    def update_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature * self.decay_rate)

    def reset_temperature(self):
        self.temperature = self.initial_temperature


class SQLiRLAgent:
    """RL Agent for SQL Injection Token Selection with Transition Masking"""

    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any] = None):
        self.state_size = state_size
        self.action_size = action_size

        # Default configuration
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'hidden_sizes': self._calculate_hidden_sizes(state_size, action_size),
            'initial_temperature': 2.0,
            'min_temperature': 0.1,
            'temperature_decay': 0.9999
        }
        self.config = {**default_config, **(config or {})}

        print(f"🧠 Neural Network Architecture:")
        print(f"   Input: {state_size} → Hidden: {self.config['hidden_sizes']} → Output: {action_size}")

        # Token handling
        self.start_tokens = ['SELECT','WITH','INSERT','UPDATE','DELETE','VALUES', "'", '"', '(', '1', '0', 'NULL', '--', '/*', '#', 'OR', 'UNION'] # Có thể điều chỉnh
        
        self.token_list = self._get_token_list()
        self.token_to_id = {token: idx for idx, token in enumerate(self.token_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.token_list)}
        self.start_token_ids = [self.token_to_id[token] for token in self.start_tokens if token in self.token_to_id]
        self.transition_table = self._build_transition_table("sqli-misc.txt")
        
        # print(f"token list value: {self.token_list}")
        # print(f"token to id value: {self.token_to_id}")
        # print(f"start token ids value: {self.start_token_ids}")

        # missing = [token for token in self.start_tokens if token not in self.token_to_id]
        # if missing:
        #     print("[WARN] Missing start tokens:", missing)
        # else:
        #     print("[INFO] All start tokens found!")s

        # Networks
        self.q_network = DQN(state_size, action_size, self.config['hidden_sizes'])
        self.target_network = DQN(state_size, action_size, self.config['hidden_sizes'])
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])

        # Exploration strategy
        self.exploration = BoltzmannExploration(
            self.config['initial_temperature'],
            self.config['min_temperature'],
            self.config['temperature_decay']
        )

        # Experience replay
        self.memory = deque(maxlen=self.config['memory_size'])
        self.step_count = 0
        self.update_target_network()

    def _calculate_hidden_sizes(self, state_size: int, action_size: int) -> List[int]:
        if action_size <= 100:
            return [512, 256, 128]
        elif action_size <= 500:
            return [1024, 512, 256]
        else:
            return [2048, 1024, 512]

    def _get_token_list(self):
        # Tùy vào project: có thể load từ file hoặc định nghĩa sẵn
        # return [str(i) for i in range(self.action_size)]
        action_space = ActionSpace("keywords.txt")
        return action_space.tokens

    def _build_transition_table(self, filename):
        """
        Xây dựng bảng chuyển tiếp token từ file sqli_misc.txt, giữ cả dấu cách là token.
        """
        table = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:

                tokens = re.findall(r"\s+|[^\s]+", line.rstrip('\n'))
                tokens = ['SPACE' if t == ' ' else t.upper() for t in tokens]
 
                for i in range(len(tokens) - 1):
                    prev_token = tokens[i]
                    next_token = tokens[i + 1]
                    if prev_token not in table:
                        table[prev_token] = []
                    table[prev_token].append(next_token)
        return table

    def select_token(self, state: np.ndarray, step_idx: int = 0, prev_token: str = None) -> int:
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

            mask = np.zeros_like(q_values)
        if step_idx == 0:
            for idx in self.start_token_ids:
                mask[idx] = 1
        elif prev_token and prev_token in self.transition_table:
            for token in self.transition_table[prev_token]:
                idx = self.token_to_id.get(token)
                if idx is not None:
                    mask[idx] = 1
        else:
            mask[:] = 1  # không match mapping → cho tất cả hợp lệ

        # Nếu không có action nào hợp lệ, fallback cho phép tất cả
        if np.sum(mask) == 0:
            mask[:] = 1

        masked_q = np.where(mask, q_values, -np.inf)
        exp_q = np.exp(masked_q - np.max(masked_q))
        probs = exp_q / np.sum(exp_q)
        action = np.random.choice(len(q_values), p=probs)
        return action


    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.q_network(state_tensor).cpu().numpy()[0]

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.config['batch_size']:
            return
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.config['target_update_freq'] == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_temperature(self):
        self.exploration.update_temperature()

    def save_model(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']

    def get_exploration_info(self) -> Dict[str, float]:
        return {
            'temperature': self.exploration.temperature,
            'min_temperature': self.exploration.min_temperature,
            'decay_rate': self.exploration.decay_rate
        }
