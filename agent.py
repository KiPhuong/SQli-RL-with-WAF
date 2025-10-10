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
from mcts_selector import MCTSSelector
import re

class DQN(nn.Module):
    """Deep Q-Network with separate Policy and Value heads."""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(DQN, self).__init__()
        # Shared body of the network
        shared_layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            shared_layers.append(nn.Linear(input_size, hidden_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        self.shared_network = nn.Sequential(*shared_layers)

        # Policy head - predicts action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(input_size, action_size),
            nn.Softmax(dim=-1)
        )

        # Value head - predicts Q-values
        self.value_head = nn.Linear(input_size, action_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        shared_features = self.shared_network(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy, value


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
            'temperature_decay': 0.9999,
            'planner': 'dqn_masked',  # 'dqn_masked' or 'mcts_lark'
            'policy_loss_weight': 0.1,
            'mcts': {
                'num_simulations': 100,
                'max_depth': 8,
                'c_puct': 1.5,
                'top_k_real': 1
            }
        }
        self.config = {**default_config, **(config or {})}

        print(f"🧠 Neural Network Architecture:")
        print(f"   Input: {state_size} → Hidden: {self.config['hidden_sizes']} → Output: {action_size}")

        # Token handling
        #self.start_tokens = ['SPACE', 'UNION'] # Có thể điều chỉnh
        self.start_tokens = ['SPACE']
        self.token_list = self._get_token_list()
        self.token_to_id = {token: idx for idx, token in enumerate(self.token_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.token_list)}
        self.start_token_ids = [self.token_to_id[token] for token in self.start_tokens if token in self.token_to_id]
        self.transition_table = self._build_transition_table("sqli-misc.txt")
        self.transition_table_noSpace = self._build_transition_table_no_space("sqli-misc.txt")

        # Precompute unions to be used as reasonable fallbacks (giữ giới hạn trong corpus thay vì 'mọi token')
        self.union_next_no_space = self._compute_union(self.transition_table_noSpace)
        self.union_next_all = self._compute_union(self.transition_table)
        
        # print(f"[DEBUG in agent] Transition table: ")
        # self.print_transition_table(self.transition_table)
        # print(f"[DEBUG in agent] Transition table noSpace: ")
        # self.print_transition_table(self.transition_table_noSpace)

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
        self.last_policy_target = None  # To store policy from MCTS

        # Initialize MCTS selector if configured
        self.mcts_selector = None
        if self.config.get('planner') == 'mcts_lark':
            try:
                # We'll need to pass the environment's evaluator function
                # For now, we'll initialize it as None and set it later
                self.mcts_selector = None  # Will be set in set_environment method
                print("   MCTS Planner enabled")
            except Exception as e:
                print(f"⚠️ Could not initialize MCTS: {e}")
                print("   Falling back to DQN masked planner")

    def set_environment(self, env):
        """Set the environment reference for MCTS evaluation."""
        self.env = env
        if self.config.get('planner') == 'mcts_lark' and env.grammar_parser:
            try:
                self.mcts_selector = MCTSSelector(
                    config=self.config,
                    grammar_parser=env.grammar_parser,
                    action_space=self.token_list,
                    env_evaluator=env._compute_potential
                )
                print("✅ MCTS Selector initialized successfully")
            except Exception as e:
                print(f"⚠️ Could not initialize MCTS: {e}")
                self.mcts_selector = None

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
        Xây dựng bảng chuyển tiếp token từ file sqli_misc.txt, giữ cả dấu cách là token,
        loại bỏ trùng lặp.
        """
        table = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                #tokens = re.findall(r"\s+|,|[^\s,]+,--|#|'.*?'|\w+|\d+|[(),]", line.rstrip('\n'))
                #tokens = ['SPACE' if t == ' ' else t.upper() for t in tokens]
                tokens = [("SPACE" if re.fullmatch(r"\s+", t) else t.upper())
                    for t in re.findall(
                    r"--.*?$|#.*?$|'|\"|\s+|\d+|[A-Za-z_][A-Za-z0-9_]*|[(),]|[^'\sA-Za-z0-9_(),]", line,flags=re.MULTILINE)
                ]
                for i in range(len(tokens) - 1):
                    prev_token = tokens[i]
                    next_token = tokens[i + 1]
                    if prev_token not in table:
                        table[prev_token] = []
                    if next_token not in table[prev_token]:
                        table[prev_token].append(next_token)
        return table
    def _build_transition_table_no_space(self, filename):
        """
        Xây dựng bảng chuyển tiếp token từ file, BỎ QUA token là SPACE.
        """
        table = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                #tokens = re.findall(r"\s+|,|[^\s,]+,--|#|'.*?'|\w+|\d+|[(),]", line.rstrip('\n'))
                tokens = [("SPACE" if re.fullmatch(r"\s+", t) else t.upper())
                    for t in re.findall(
                    r"--.*?$|#.*?$|'|\"|\s+|\d+|[A-Za-z_][A-Za-z0-9_]*|[(),]|[^'\sA-Za-z0-9_(),]", line,flags=re.MULTILINE)
                ]
                # Bỏ qua token là dấu cách
                #tokens = [t.upper() for t in tokens if t != ' ']
                tokens = [t for t in tokens if t != "SPACE"]
                for i in range(len(tokens) - 1):
                    prev_token = tokens[i]
                    next_token = tokens[i + 1]
                    if prev_token not in table:
                        table[prev_token] = []
                    if next_token not in table[prev_token]:
                        table[prev_token].append(next_token)
        return table
        

    def print_transition_table(self, table):
        print(f"{'Prev Token':<25} {'Next Token':<15}")
        print("-" * 40)
        for prev_token, next_tokens in table.items():
            for next_token in next_tokens:
                print(f"{prev_token:<25} {next_token:<15}")

    # def select_token(self, state: np.ndarray, step_idx: int = 0, prev_token: str = None) -> int:
    #     self.q_network.eval()
    #     with torch.no_grad():
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #         q_values = self.q_network(state_tensor).cpu().numpy()[0]

    #     mask = np.zeros_like(q_values)

    #     if step_idx == 0:
    #         for idx in self.start_token_ids:
    #             mask[idx] = 1
    #     elif prev_token and prev_token in self.transition_table:
    #         for token in self.transition_table[prev_token]:
    #             idx = self.token_to_id.get(token)
    #             if idx is not None:
    #                 mask[idx] = 1
    #     else:
    #         mask[:] = 1  # không match mapping → cho tất cả hợp lệ

    #     # Nếu không có action nào hợp lệ, fallback cho phép tất cả
    #     if np.sum(mask) == 0:
    #         mask[:] = 1

    #     masked_q = np.where(mask, q_values, -np.inf)
    #     exp_q = np.exp(masked_q - np.max(masked_q))
    #     probs = exp_q / np.sum(exp_q)
    #     action = np.random.choice(len(q_values), p=probs)

    #     return action

    def select_token(self, state: np.ndarray, step_idx: int = 0, prev_token: str = None, prev_prev_token: str = None, current_payload: str = "") -> int:
        # If MCTS planner is enabled, use it
        if self.config.get('planner') == 'mcts_lark' and self.mcts_selector:
            # 1. Get Q-values from DQN to use as policy priors
            q_values = self.get_q_values(state)

            # 2. Get a mask of allowed actions to ensure priors are valid
            allowed_ids = self._get_allowed_ids(step_idx, prev_token, prev_prev_token)
            mask_bool = np.zeros(self.action_size, dtype=bool)
            mask_bool[list(allowed_ids)] = True

            # 3. Convert Q-values to probabilities (softmax) to guide MCTS
            probabilities = self._masked_softmax(q_values, mask_bool)
            policy_priors = {self.id_to_token[i]: prob for i, prob in enumerate(probabilities) if prob > 0}

            # 4. Run MCTS search with DQN guidance
            best_action_token, policy_target = self.mcts_selector.select_action(
                initial_payload=current_payload,
                policy_priors=policy_priors
            )
            action_id = self.token_to_id.get(best_action_token, -1)

            # Store policy target for later use in training
            self.last_policy_target = policy_target
            if action_id != -1:
                return action_id
            else:
                # Fallback if token from MCTS is not in our vocab (should be rare)
                print(f"[WARN] MCTS selected token '{best_action_token}' not in vocab. Falling back to DQN.")

        # Fallback to original DQN masked softmax logic
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, q_values = self.q_network(state_tensor)
            q_values = q_values.cpu().numpy()[0]

        mask = np.zeros_like(q_values)

        if step_idx == 0:
            for idx in self.start_token_ids:
                mask[idx] = 1
        elif prev_token == "SPACE" and prev_prev_token in self.transition_table_noSpace:
            # Nếu prev_token là SPACE, dùng transition_table_noSpace
            for token in self.transition_table_noSpace[prev_prev_token]:
                idx = self.token_to_id.get(token)
                if idx is not None:
                    mask[idx] = 1
        elif prev_token and prev_token in self.transition_table:
            for token in self.transition_table[prev_token]:
                idx = self.token_to_id.get(token)
                if idx is not None:
                    mask[idx] = 1
        else:
            mask[:] = 1  # không match mapping → cho tất cả hợp lệ

        if np.sum(mask) == 0:
            mask[:] = 1

        # masked_q = np.where(mask, q_values, -np.inf)
        # exp_q = np.exp(masked_q - np.max(masked_q))
        # probs = exp_q / np.sum(exp_q)

        masked_q = np.full_like(q_values, -np.inf)
        masked_q[mask.astype(bool)] = q_values[mask.astype(bool)]

        # tránh nan
        if np.all(~np.isfinite(masked_q)):
            masked_q = q_values  # fallback: cho phép tất cả

        logits = (masked_q - np.nanmax(masked_q)) / max(self.exploration.temperature, 1e-6)
        exp_q = np.exp(logits)
        probs = exp_q / np.sum(exp_q)

        action = np.random.choice(len(q_values), p=probs)
        # print(f"[DEBUG in agent] Q-values: \n {q_values}")
        # print(f"[DEBUG in agent] Probs Action: \n {probs}")
        # print(f"[DEBUG in agent] Prev_prev_token: {str(prev_prev_token):<10} || Prve_token: {str(prev_token):<10} || Action selected: {self.id_to_token.get(action)}")
        return action

        # --- helper: compute union of next tokens from a transition table ---
    def _compute_union(self, table: Dict[str, List[str]]) -> set:
        s = set()
        for lst in table.values():
            s.update(lst)
        return s

    def _tokens_to_ids(self, tokens_set: set) -> set:
        ids = set()
        for tok in tokens_set:
            idx = self.token_to_id.get(tok)
            if idx is not None:
                ids.add(idx)
        return ids

    def _get_allowed_ids(self, step_idx: int, prev_token: str = None, prev_prev_token: str = None) -> set:
        """
        Trả về set các action ids hợp lệ dựa trên step_idx, prev_token, prev_prev_token.
        Quy tắc:
         - step_idx == 0: trả về start_token_ids (nếu rỗng -> fallback cấp phép tất cả nhưng warn)
         - nếu prev_token == 'SPACE': nếu có prev_prev_token -> xem transition_table_noSpace[prev_prev_token]
                                       ngược lại -> union_next_no_space (fallback, hạn chế trong corpus)
         - nếu prev_token in transition_table -> dùng transition_table[prev_token]
         - else -> fallback union_next_all (tập token xuất hiện trong corpus)
        """
        # Begin
        if step_idx == 0:
            allowed_ids = set(self.start_token_ids)
            if len(allowed_ids) == 0:
                # cực hiếm — cho phép tất cả nhưng báo
                print("[WARN] start_token_ids empty — allowing all tokens as fallback")
                return set(range(self.action_size))
            return allowed_ids

        # case prev_token is SPACE (we want next token that follows the token before SPACE)
        if prev_token == "SPACE":
            if prev_prev_token is not None and prev_prev_token in self.transition_table_noSpace:
                allowed_tokens = set(self.transition_table_noSpace[prev_prev_token])
            else:
                # fallback: union of tokens that ever follow non-space tokens (hạn chế trong corpus)
                allowed_tokens = set(self.union_next_no_space)
        elif prev_token is not None and prev_token in self.transition_table:
            allowed_tokens = set(self.transition_table[prev_token])
        else:
            # fallback: union of all next tokens observed in corpus
            allowed_tokens = set(self.union_next_all)
            # additionally include start tokens (optional)
            allowed_tokens.update([self.id_to_token[i] for i in self.start_token_ids if i in self.id_to_token])

        allowed_ids = self._tokens_to_ids(allowed_tokens)
        if not allowed_ids:
            # nếu vẫn không có token hợp lệ, fallback an toàn: cho phép tất cả
            print("[WARN] Allowed token ids empty -> fallback to ALL actions")
            allowed_ids = set(range(self.action_size))
        return allowed_ids

    def _masked_softmax(self, q_values: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        """
        Softmax có mask, dùng temperature self.exploration.temperature và fallback nếu cần.
        - q_values: shape (action_size,)
        - mask_bool: boolean np array shape (action_size,) True=allowed
        """
        # Mảng logits: -inf ở chỗ không được phép
        masked_q = np.full_like(q_values, -np.inf, dtype=float)
        if mask_bool.any():
            masked_q[mask_bool] = q_values[mask_bool]
        else:
            # fallback: allow all (rare)
            masked_q = q_values.copy()

        # kiểm tra phần tử hữu hạn
        finite_mask = np.isfinite(masked_q)
        if not finite_mask.any():
            # không có giá trị hữu hạn → fallback uniform trên mask (nếu mask bool any) hoặc uniform global
            if mask_bool.any():
                probs = mask_bool.astype(float) / float(mask_bool.sum())
            else:
                probs = np.ones_like(q_values, dtype=float) / float(len(q_values))
            return probs

        # numeric stability: lấy max trên các giá trị hữu hạn
        max_val = float(np.max(masked_q[finite_mask]))
        T = max(self.exploration.temperature, 1e-6)
        scaled = (masked_q - max_val) / T

        exp_vals = np.zeros_like(scaled, dtype=float)
        exp_vals[finite_mask] = np.exp(scaled[finite_mask])

        sum_exp = float(np.sum(exp_vals))
        if not np.isfinite(sum_exp) or sum_exp <= 0:
            # fallback: uniform over mask_bool
            if mask_bool.any():
                probs = mask_bool.astype(float) / float(mask_bool.sum())
            else:
                probs = np.ones_like(q_values, dtype=float) / float(len(q_values))
            return probs

        probs = exp_vals / sum_exp
        return probs


    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        policy_target = self.last_policy_target or {}
        self.memory.append((state, action, reward, next_state, done, policy_target))
        self.last_policy_target = None  # Reset after use

    def replay(self):
        if len(self.memory) < self.config['batch_size']:
            return
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones, policy_targets = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Convert policy targets (dicts) to a dense tensor
        policy_target_tensor = torch.zeros(self.config['batch_size'], self.action_size)
        for i, target_dict in enumerate(policy_targets):
            if target_dict: # MCTS might not return targets if no children
                for token, prob in target_dict.items():
                    if token in self.token_to_id:
                        policy_target_tensor[i, self.token_to_id[token]] = prob

        # Get current policy and value predictions
        current_policy_preds, current_q_values_all = self.q_network(states)
        current_q_values = current_q_values_all.gather(1, actions.unsqueeze(1))

        # Get target Q-values
        _, next_q_values_all = self.target_network(next_states)
        next_q_values = next_q_values_all.max(1)[0].detach()
        target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)

        # 1. Value Loss (MSE)
        value_loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 2. Policy Loss (Cross-Entropy)
        # We only calculate policy loss for samples that came from MCTS
        mcts_samples_mask = [bool(p) for p in policy_targets]
        if any(mcts_samples_mask):
            policy_loss = nn.CrossEntropyLoss()(current_policy_preds[mcts_samples_mask], policy_target_tensor[mcts_samples_mask])
        else:
            policy_loss = torch.tensor(0.0) # No MCTS samples in this batch

        # 3. Combined Loss
        loss = value_loss + self.config.get('policy_loss_weight', 0.1) * policy_loss

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
