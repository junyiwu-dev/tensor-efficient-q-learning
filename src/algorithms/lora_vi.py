import numpy as np
import time

def khatri_rao(matrices):
    """
    Compute the Khatri-Rao product (column-wise Kronecker product) of matrices.
    
    Args:
        matrices: List of 2D numpy arrays
        
    Returns:
        Khatri-Rao product
    """
    if len(matrices) == 0:
        raise ValueError("Need at least one matrix")
    
    if len(matrices) == 1:
        return matrices[0]
    
    # Start with the first matrix
    result = matrices[0]
    
    # Iteratively compute Khatri-Rao product
    for matrix in matrices[1:]:
        # Get dimensions
        n_row_res, n_col = result.shape
        n_row_mat, n_col_check = matrix.shape
        
        if n_col != n_col_check:
            raise ValueError(f"All matrices must have the same number of columns. Got {n_col} and {n_col_check}")
        
        # Compute Khatri-Rao product
        new_result = np.zeros((n_row_res * n_row_mat, n_col))
        for col in range(n_col):
            new_result[:, col] = np.kron(result[:, col], matrix[:, col])
        
        result = new_result
    
    return result

class LoRaVI:
    """
    Low-Rank Value Iteration (LoRa-VI) implementation
    Based on "Model-free Low-Rank Reinforcement Learning via Leveraged Entry-wise Matrix Estimation"
    
    This is a simplified version that maintains compatibility with the existing tensor-based framework.
    """
    def __init__(
        self,
        env,
        discretizer,
        episodes,
        max_steps,
        epsilon,        
        alpha,
        gamma,
        k,
        decay=1.0,      
        decay_alpha=1.0,
        init_ord=1,
        min_epsilon=0.0,  
        bias=0.0,
        normalize_columns=False,
        convergence_threshold=0.01,
        max_inner_iterations=5,
        policy_improvement_freq=10,  # How often to perform policy improvement
        leverage_sample_ratio=0.5,   # Ratio of samples used for leverage scores estimation
    ):
        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.decay_alpha = decay_alpha
        self.min_epsilon = min_epsilon
        self.k = k
        self.normalize_columns = normalize_columns
        self.convergence_threshold = convergence_threshold
        self.max_inner_iterations = max_inner_iterations
        self.policy_improvement_freq = policy_improvement_freq
        self.leverage_sample_ratio = leverage_sample_ratio
        self.current_episode = 0

        # Initialize factor matrices (low-rank tensor decomposition)
        self.factors = [(np.random.rand(dim, self.k) - bias) * init_ord 
                       for dim in self.discretizer.dimensions]
        self.factor_indices = np.arange(len(self.factors))
        self.non_factor_indices = [np.delete(self.factor_indices, i).tolist() 
                                  for i in self.factor_indices]

        # Initialize visit counts
        q_shape = tuple(self.discretizer.dimensions)
        self.N = np.zeros(q_shape, dtype=np.float64)
        
        # Store leverage scores (simplified version)
        self.row_leverage = np.ones(np.prod(self.discretizer.n_states)) / np.prod(self.discretizer.n_states)
        self.col_leverage = np.ones(np.prod(self.discretizer.n_actions)) / np.prod(self.discretizer.n_actions)
        
        # Current policy (deterministic)
        state_dims = np.prod(self.discretizer.n_states)
        action_dims = np.prod(self.discretizer.n_actions)
        self.policy = np.zeros(state_dims, dtype=np.int32)
        
        # Training history
        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.action_history = []
        self.state_history = []

    def get_random_action(self):
        """Sample a random action from the action space"""
        return np.random.uniform(self.discretizer.min_points_actions, 
                                self.discretizer.max_points_actions)

    def get_greedy_action(self, state):
        """Get greedy action based on current Q-function estimate"""
        state_idx = self.discretizer.get_state_index(state)
        q_values = self.get_q_from_state_idx(state_idx).reshape(self.discretizer.n_actions)
        action_idx = np.unravel_index(np.argmax(q_values), q_values.shape)
        return self.discretizer.get_action_from_index(action_idx)

    def get_policy_action(self, state):
        """Get action from current policy"""
        state_idx = self.discretizer.get_state_index(state)
        flat_state_idx = np.ravel_multi_index(state_idx, self.discretizer.n_states)
        flat_action_idx = self.policy[flat_state_idx]
        action_idx = np.unravel_index(flat_action_idx, self.discretizer.n_actions)
        return self.discretizer.get_action_from_index(action_idx)

    def get_q_from_state_idx(self, state_idx):
        """Compute Q-values for all actions given a state index"""
        state_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors[:len(state_idx)]):
            factor_row = factor[state_idx[idx], :]
            state_value *= factor_row
        action_khatri = khatri_rao(self.factors[len(state_idx):])
        action_state = np.dot(state_value, action_khatri.T)
        return action_state

    def get_q_from_state_action_idx(self, state_idx, action_idx):
        """Compute Q-value for a specific state-action pair"""
        indices = state_idx + action_idx
        q_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors):
            q_value *= factor[indices[idx], :]
        return np.sum(q_value)

    def choose_action(self, state, use_policy=False, epsilon_greedy=True):
        """
        Choose action based on current strategy
        
        Args:
            state: Current state
            use_policy: If True, use current policy; otherwise use Q-function
            epsilon_greedy: If True, use epsilon-greedy; otherwise purely greedy
        """
        if epsilon_greedy and np.random.random() < self.epsilon:
            return self.get_random_action()

        if use_policy:
            return self.get_policy_action(state)
        else:
            return self.get_greedy_action(state)

    def update_leverage_scores(self):
        """
        Simplified leverage scores update based on SVD of factor matrices
        In full LoRa-VI, this would use the complete LME Phase 1 procedure
        """
        # Combine state factors
        if len(self.factors[:len(self.discretizer.n_states)]) > 1:
            state_factors = np.kron(*[f for f in self.factors[:len(self.discretizer.n_states)]])
        else:
            state_factors = self.factors[0]
        
        # Combine action factors  
        if len(self.factors[len(self.discretizer.n_states):]) > 1:
            action_factors = np.kron(*[f for f in self.factors[len(self.discretizer.n_states):]])
        else:
            action_factors = self.factors[len(self.discretizer.n_states)]
        
        # Compute leverage scores from factor matrices
        # Leverage scores are proportional to row norms of left/right singular vectors
        try:
            U, _, _ = np.linalg.svd(state_factors, full_matrices=False)
            row_scores = np.sum(U[:, :self.k]**2, axis=1)
            self.row_leverage = row_scores / np.sum(row_scores)
        except:
            # If SVD fails, use uniform distribution
            self.row_leverage = np.ones(state_factors.shape[0]) / state_factors.shape[0]
        
        try:
            _, _, Vt = np.linalg.svd(action_factors, full_matrices=False)
            col_scores = np.sum(Vt[:self.k, :]**2, axis=0)
            self.col_leverage = col_scores / np.sum(col_scores)
        except:
            self.col_leverage = np.ones(action_factors.shape[0]) / action_factors.shape[0]

    def policy_evaluation_step(self, state, action, state_prime, reward, done):
        """
        Update Q-function using low-rank tensor update (Policy Evaluation)
        This is a simplified version of LME Phase 2
        """
        state_idx = self.discretizer.get_state_index(state)
        action_idx = self.discretizer.get_action_index(action)
        indices = state_idx + action_idx

        # Update visit counts
        self.N[indices] += 1

        # Compute TD target
        if not done:
            state_prime_idx = self.discretizer.get_state_index(state_prime)
            q_prime = self.get_q_from_state_idx(state_prime_idx).reshape(self.discretizer.n_actions)
            max_q_prime = np.max(q_prime)
            target = reward + self.gamma * max_q_prime
        else:
            target = reward

        # Current Q-value
        current_q = self.get_q_from_state_action_idx(state_idx, action_idx)

        # TD error
        td_error = target - current_q

        # Update factor matrices using block coordinate descent
        for factor_idx in range(len(self.factors)):
            # Compute gradient for this factor
            grad_factor = np.zeros_like(self.factors[factor_idx][indices[factor_idx], :])
            
            # Compute product of all other factors
            other_factors_prod = np.ones(self.k)
            for j, factor in enumerate(self.factors):
                if j != factor_idx:
                    other_factors_prod *= factor[indices[j], :]
            
            # Gradient descent update with frequency-based regularization
            weight = 1.0 / (self.N[indices] + 1.0)**0.5  # Inverse frequency weighting
            grad_factor = -2 * weight * td_error * other_factors_prod
            
            # Update with learning rate
            self.factors[factor_idx][indices[factor_idx], :] -= self.alpha * grad_factor

    def policy_improvement(self):
        """
        Update policy to be greedy with respect to current Q-function
        """
        state_dims = np.prod(self.discretizer.n_states)
        
        for flat_state_idx in range(state_dims):
            # Convert flat index to multi-dimensional index
            state_idx = np.unravel_index(flat_state_idx, self.discretizer.n_states)
            
            # Get Q-values for all actions
            q_values = self.get_q_from_state_idx(state_idx).reshape(self.discretizer.n_actions)
            
            # Update policy to select action with highest Q-value
            flat_action_idx = np.argmax(q_values)
            self.policy[flat_state_idx] = flat_action_idx

    def run_episode(self, is_train=True, is_greedy=False, use_policy=False):
        """
        Run one episode of interaction with the environment
        
        Args:
            is_train: Whether to update Q-function
            is_greedy: Whether to use greedy action selection
            use_policy: Whether to follow current policy (for policy iteration)
        """
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        if len(state.shape) > 1:
            state = state.flatten()

        cumulative_reward = 0
        episode_states = []
        episode_actions = []
        steps = 0

        for step in range(self.max_steps):
            # Choose action
            if use_policy:
                action = self.choose_action(state, use_policy=True, epsilon_greedy=not is_greedy)
            else:
                action = self.choose_action(state, use_policy=False, epsilon_greedy=not is_greedy)

            if self.discretizer.discrete_action:
                action = int(np.round(action))
                episode_actions.append(action)
            else:
                if isinstance(action, np.ndarray):
                    episode_states.append(state.copy())
                    episode_actions.append(action.copy())

            # Take action (handle both old and new gym API)
            step_result = self.env.step(action)
            if len(step_result) == 5:
                # New gym API (>= 0.26): state, reward, terminated, truncated, info
                state_prime, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Old gym API (< 0.26): state, reward, done, info
                state_prime, reward, done, info = step_result
            
            if isinstance(info, dict) and 'terminal_observation' in info:
                done = done[0] if isinstance(done, tuple) else done

            cumulative_reward += reward
            if len(state_prime.shape) > 1:
                state_prime = state_prime.flatten()

            # Update Q-function if training
            if is_train:
                self.policy_evaluation_step(state, action, state_prime, reward, done)

            if done:
                steps = step + 1
                break

            state = state_prime
            
            # Decay epsilon and alpha
            if (not is_greedy) and is_train and (self.epsilon > self.min_epsilon):
                self.epsilon *= self.decay
            if not is_greedy:
                self.alpha *= self.decay_alpha

        self.state_history.append(episode_states)
        self.action_history.append(episode_actions)
        steps = steps or self.max_steps
        
        if is_train and not is_greedy:
            self.current_episode += 1

        return steps, cumulative_reward

    def train(self, run_greedy_frequency=None):
        """
        Main training loop for LoRa-VI
        Alternates between policy evaluation and policy improvement
        """
        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.current_episode = 0
        
        for episode in range(self.episodes):
            # Run training episode (policy evaluation)
            steps, episode_reward = self.run_episode(is_train=True, is_greedy=False, use_policy=False)
            self.training_steps.append(steps)
            self.training_cumulative_reward.append(episode_reward)
            
            # Periodically perform policy improvement
            if episode > 0 and (episode % self.policy_improvement_freq == 0):
                # Update leverage scores (simplified LME Phase 1)
                self.update_leverage_scores()
                
                # Policy improvement (greedy update)
                self.policy_improvement()
            
            # Run greedy evaluation episode
            if run_greedy_frequency and (episode % run_greedy_frequency == 0):
                greedy_steps, greedy_reward = self.run_episode(is_train=False, is_greedy=True, use_policy=True)
                self.greedy_steps.append(greedy_steps)
                self.greedy_cumulative_reward.append(greedy_reward)
                # Print greedy evaluation results
                print(f"Episode {episode:4d} [Greedy] - Steps: {greedy_steps:4d}, Reward: {float(greedy_reward):8.2f}")

        self.evaluate_final_policy()

    def evaluate_final_policy(self):
        """Evaluate final policy performance"""
        rewards = []
        for _ in range(1000):
            _, cumulative_reward = self.run_episode(is_train=False, is_greedy=True, use_policy=True)
            rewards.append(cumulative_reward)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)

    def save_policy(self, filepath):
        """Save the learned policy"""
        policy_data = {
            'factors': [f.tolist() for f in self.factors],
            'policy': self.policy.tolist(),
            'parameters': {
                'k': self.k,
                'gamma': self.gamma,
                'mean_reward': getattr(self, 'mean_reward', None),
                'std_reward': getattr(self, 'std_reward', None)
            }
        }
        import json
        with open(filepath, 'w') as f:
            json.dump(policy_data, f)