"""
Monte Carlo Tree Search (MCTS) Selector for SQLi RL Agent
"""

import math
import random
from typing import Dict, Any, Optional, List
from lark import Lark, UnexpectedToken

from macro_actions import MacroActionManager
class MCTSNode:
    """A node in the MCTS search tree."""
    def __init__(self, payload: str, parent: Optional['MCTSNode'] = None, action_token: Optional[str] = None, prior_prob: float = 0.0):
        self.payload = payload
        self.parent = parent
        self.action_token = action_token  # The token that led to this node
        self.prior_prob = prior_prob

        self.children: List['MCTSNode'] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.is_fully_expanded = False

    @property
    def q_value(self) -> float:
        """Return the average value of this node (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

class MCTSSelector:
    """Manages the MCTS process to select the best action."""
    def __init__(self, config: Dict[str, Any], grammar_parser: Any, action_space: List[str], env_evaluator: Any):
        self.config = config
        self.grammar_parser = grammar_parser
        self.action_space = action_space
        self.env_evaluator = env_evaluator # Function to evaluate a payload (e.g., env._compute_potential)

        self.c_puct = config.get('mcts', {}).get('c_puct', 1.5)
        self.num_simulations = config.get('mcts', {}).get('num_simulations', 100)
        self.top_k_real = config.get('mcts', {}).get('top_k_real', 1)

        # Initialize macro action manager
        self.macro_manager = MacroActionManager(config)

        # Cache for HTTP evaluations
        self.evaluation_cache = {}

    def select_action(self, initial_payload: str, policy_priors: Dict[str, float] = None) -> tuple:
        """Run the MCTS search and return the best action token and policy target."""
        root = MCTSNode(payload=initial_payload)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_fully_expanded:
                node = self._expand(node, policy_priors)

            reward = self._simulate(node)
            self._backpropagate(node, reward)

        best_child = self._get_best_child(root, exploration_weight=0.0) # Get best child for exploitation
        best_action_token = best_child.action_token if best_child else random.choice(self.action_space)

        policy_target = self._get_policy_target(root)

        return best_action_token, policy_target

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using the UCT formula."""
        while node.children:
            if not node.is_fully_expanded:
                return node
            node = self._get_best_child(node, exploration_weight=self.c_puct)
        return node

    def _expand(self, node: MCTSNode, policy_priors: Dict[str, float] = None) -> MCTSNode:
        """Expand the selected node by creating one new child."""
        # Get both individual tokens and macro actions
        valid_tokens = self._get_valid_actions(node.payload)
        applicable_macros = self.macro_manager.get_applicable_macros(node.payload)

        # Combine individual tokens and macro patterns
        all_actions = valid_tokens + [macro.pattern for macro in applicable_macros]

        for action in all_actions:
            # Check if a child for this action already exists
            if not any(child.action_token == action for child in node.children):
                new_payload = self._construct_new_payload(node.payload, action)
                prior = policy_priors.get(action, 0.0) if policy_priors else 0.0
                child_node = MCTSNode(payload=new_payload, parent=node, action_token=action, prior_prob=prior)
                node.children.append(child_node)
                return child_node

        node.is_fully_expanded = True
        return node # Should not be reached if selection logic is correct

    def _simulate(self, node: MCTSNode) -> float:
        """Simulate the outcome from the node, returning a reward."""
        # Check cache first
        if node.payload in self.evaluation_cache:
            return self.evaluation_cache[node.payload]

        # Use lightweight evaluation (potential-based) for most nodes
        reward = self.env_evaluator(node.payload)

        # Cache the result
        self.evaluation_cache[node.payload] = reward
        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Propagate the simulation result back up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def _get_best_child(self, node: MCTSNode, exploration_weight: float = 0.0) -> Optional[MCTSNode]:
        """Select the best child based on the PUCT formula."""
        if not node.children:
            return None

        best_score = -float('inf')
        best_children = []

        for child in node.children:
            if child.visit_count == 0 and exploration_weight > 0:
                score = float('inf') # Prioritize unvisited nodes
            else:
                # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
                exploit_term = child.q_value
                explore_term = exploration_weight * child.prior_prob * (math.sqrt(node.visit_count) / (1 + child.visit_count))
                score = exploit_term + explore_term

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children) if best_children else None

    def _get_policy_target(self, root: MCTSNode) -> Dict[str, float]:
        """Generate policy target from visit counts of root's children."""
        if not root.children:
            return {}

        # Get visit counts for all children
        visit_counts = {child.action_token: child.visit_count for child in root.children}
        total_visits = sum(visit_counts.values())

        # Normalize to get probabilities
        if total_visits == 0:
            return {}

        policy_target = {action: count / total_visits for action, count in visit_counts.items()}
        return policy_target

    def _get_valid_actions(self, payload: str) -> List[str]:
        """Get valid next actions based on the grammar using an interactive parser."""
        if not self.grammar_parser:
            return self.action_space # Fallback if grammar is not available

        interactive_parser = self.grammar_parser.parse_interactive(payload)
        try:
            interactive_parser.exhaust_lexer()
            accepted_tokens = interactive_parser.accepts()
            # Filter out special Lark tokens and return only valid token names
            valid_actions = [token for token in accepted_tokens if token in self.action_space]
            return valid_actions if valid_actions else [" "] # Fallback to space if no other options
        except UnexpectedToken:
            # If the payload is invalid, we can't suggest next tokens
            return [" "] # Fallback to allow recovery

    def _construct_new_payload(self, current_payload: str, new_token: str) -> str:
        """Construct new payload by intelligently joining tokens."""
        if not current_payload:
            return new_token

        # Handle special tokens
        if new_token == 'SPACE':
            return current_payload + ' '

        # No space needed before/after certain characters
        no_space_before = ['(', ')', ',', ';', '.', '=', '!=', '<', '>', '<=', '>=']
        no_space_after = ['(', '.', '=', '!=', '<', '>', '<=', '>=']

        last_char = current_payload[-1] if current_payload else ''

        # Check if we need space
        need_space = True

        if new_token in no_space_before or last_char in no_space_after:
            need_space = False

        # Special cases for quotes and comments
        if new_token in ["'", '"'] or last_char in ["'", '"']:
            need_space = False

        if new_token.startswith('--') or new_token.startswith('/*'):
            need_space = True

        # Construct new payload
        if need_space:
            return current_payload + ' ' + new_token
        else:
            return current_payload + new_token

