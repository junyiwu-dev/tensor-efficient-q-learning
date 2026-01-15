import numpy as np
import time
import tensorly as tl
import torch
import matplotlib.pyplot as plt

torch.set_num_threads(1)

class TensorEfficientQL:
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
        ucb_c=1.0,       
        lambda_penalty=1e-5,
        epsilon_penalty=1e-4,  
        tau_start = 1.0,
        tau_end = 0.05, 
        tau_decay_episodes = 5000
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
        self.ucb_c = ucb_c
        self.lambda_penalty = lambda_penalty
        self.epsilon_penalty = epsilon_penalty
        self.current_episode = 0
        self.tau_start = tau_start
        self.tau_end = tau_end 
        self.tau_decay_episodes = tau_decay_episodes  

        self.factors = [(np.random.rand(dim, self.k) - bias) * init_ord for dim in self.discretizer.dimensions]
        self.factor_indices = np.arange(len(self.factors))
        self.non_factor_indices = [np.delete(self.factor_indices, i).tolist() for i in self.factor_indices]

        q_error_shape = tuple(self.discretizer.dimensions)
        self.Q_error = np.zeros(q_error_shape)
        self.N = np.zeros(q_error_shape, dtype=np.float64)  
        self.N_total = np.zeros(tuple(self.discretizer.n_states), dtype=np.float64)  

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.action_history = []
        self.state_history = []
        # Print all hyperparameters
        self._print_hyperparameters()

    def _print_hyperparameters(self):
        """Print all hyperparameter configurations"""
        print("\n" + "="*70)
        print("TEQL Hyperparameters Configuration")
        print("="*70)
        print(f"Environment Parameters:")
        print(f"  episodes:             {self.episodes}")
        print(f"  max_steps:            {self.max_steps}")
        print(f"\nRL Core Parameters:")
        print(f"  epsilon (initial):    {self.epsilon}")
        print(f"  min_epsilon:          {self.min_epsilon}")
        print(f"  alpha (learning rate):{self.alpha}")
        print(f"  gamma (discount):     {self.gamma}")
        print(f"  decay (epsilon):      {self.decay}")
        print(f"  decay_alpha:          {self.decay_alpha}")
        print(f"\nTensor Decomposition:")
        print(f"  k (rank):             {self.k}")
        print(f"  normalize_columns:    {self.normalize_columns}")
        print(f"\nConvergence Control:")
        print(f"  convergence_threshold:{self.convergence_threshold}")
        print(f"  max_inner_iterations: {self.max_inner_iterations}")
        print(f"\nExploration (EUGE):")
        print(f"  ucb_c:                {self.ucb_c}")
        print(f"\nRegularization:")
        print(f"  lambda_penalty:       {self.lambda_penalty}")
        print(f"  epsilon_penalty:      {self.epsilon_penalty}")
        print(f"\nDiscretization:")
        print(f"  state dimensions:     {self.discretizer.n_states}")
        print(f"  action dimensions:    {self.discretizer.n_actions}")
        print(f"  total tensor shape:   {self.discretizer.dimensions}")
        print("="*70 + "\n")

    def get_random_action(self):
        """Same as TLR: use environment's action_space to sample random actions"""
        return self.env.action_space.sample()

    def get_greedy_action(self, state):
        """Same as TLR greedy strategy, supports discrete and continuous actions"""
        state_idx = self.discretizer.get_state_index(state)
        q_simplified = self.get_q_from_state_idx(state_idx).reshape(self.discretizer.n_actions)
        action_idx = np.unravel_index(np.argmax(q_simplified), q_simplified.shape)

        # Discrete action: return integer index directly; Continuous action: map back to actual action via discretizer
        if getattr(self.discretizer, "discrete_action", False):
            return action_idx[0]
        return self.discretizer.get_action_from_index(action_idx)

    def get_q_from_state_idx(self, state_idx):
        """Use the same Khatri-Rao expansion order and computation method as TLR"""
        state_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors[:len(state_idx)]):
            state_value *= factor[state_idx[idx], :]

        # Key point: explicitly specify reverse=False to ensure action dimension order is consistent with TLR
        action_khatri = tl.tenalg.khatri_rao(self.factors[len(state_idx):], reverse=False)

        # Same matrix operation form as TLR
        action_state = np.sum(state_value * action_khatri, axis=1)
        return action_state

    def get_q_from_state_action_idx(self, state_idx, action_idx):
        indices = state_idx + action_idx
        q_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors):
            q_value *= factor[indices[idx], :]
        return np.sum(q_value)
    
    def choose_action(self, state):
  
        if np.random.random() < self.epsilon:
            return self.get_random_action()

        state_idx = self.discretizer.get_state_index(state)
        
        if self.discretizer.discrete_action:
            possible_actions = list(range(self.discretizer.n_actions[0]))
        else:
            action_shape = tuple(self.discretizer.n_actions)
            possible_action_indices = np.indices(action_shape).reshape(len(action_shape), -1).T
            possible_actions = [tuple(idx) for idx in possible_action_indices]

        ucb_scores = []
        for action_idx in possible_actions:
            if self.discretizer.discrete_action:
                action_idx = (action_idx,)
            q_a = self.get_q_from_state_action_idx(state_idx, action_idx)
            error_bonus = self.Q_error[state_idx + action_idx]
            n_sa = self.N[state_idx + action_idx]
            n_s_total = self.N_total[state_idx] if self.N_total[state_idx] > 0 else 1
            count_bonus = np.sqrt(np.log(n_s_total) / (n_sa + 1))
            ucb_a = q_a + self.ucb_c * (error_bonus + count_bonus)
            ucb_scores.append(ucb_a)

        chosen_action_idx = np.argmax(ucb_scores)
        chosen_action = possible_actions[chosen_action_idx]
        return self.discretizer.get_action_from_index(chosen_action)
    
    
    def normalize(self):
        n_factors = len(self.factors)
        power_denominator = (n_factors - 1) / n_factors
        power_numerator = 1 / n_factors
        norms_denominator = [np.linalg.norm(factor, axis=0)**power_denominator for factor in self.factors]
        norms_numerator = [np.linalg.norm(factor, axis=0)**power_numerator for factor in self.factors]
        for i in range(n_factors):
            numerator = np.prod([norms_numerator[j] for j in range(n_factors) if j != i], axis=0)
            scaler = numerator / norms_denominator[i]
            self.factors[i] *= scaler

    def update_q_matrix(self, state, action, state_prime, reward, done):
        """Update low-rank Q matrix.
        - When lambda_penalty==0 and max_inner_iterations==1: equivalent to TLR;
        - Otherwise: use TEQL iterative update with optional penalty.
        """
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        q_next = np.max(self.get_q_from_state_idx(state_prime_idx)) if not done else 0.0
        target_q = reward + self.gamma * q_next

        tensor_indices = state_idx + action_idx
        q_before = self.get_q_from_state_action_idx(state_idx, action_idx)

        n_sa = self.N[tensor_indices]
        penalty_weight = self.lambda_penalty / (n_sa + self.epsilon_penalty)

        iteration = 0
        q_prev = q_before

        while iteration < self.max_inner_iterations:
            new_factors = [factor.copy() for factor in self.factors]
            for factor_idx in self.factor_indices:
                grad_factor = np.ones(self.k)
                for non_factor_idx in self.non_factor_indices[factor_idx]:
                    grad_factor *= self.factors[non_factor_idx][tensor_indices[non_factor_idx], :]

                grad_norm = np.linalg.norm(grad_factor)
                if grad_norm == 0.0:
                    continue

                q_current = self.get_q_from_state_action_idx(state_idx, action_idx)
                error_signal = target_q - q_current

                error_update = -error_signal * grad_factor / grad_norm

                if self.lambda_penalty != 0.0:
                    encouragement_update = penalty_weight * grad_factor / grad_norm
                else:
                    encouragement_update = 0.0

                update = error_update - encouragement_update
                new_factors[factor_idx][tensor_indices[factor_idx], :] -= self.alpha * update

            self.factors = new_factors
            if self.normalize_columns:
                self.normalize()

            q_new = self.get_q_from_state_action_idx(state_idx, action_idx)
            if abs(q_new - q_prev) < self.convergence_threshold:
                break
            q_prev = q_new
            iteration += 1

        q_after = self.get_q_from_state_action_idx(state_idx, action_idx)
        self.Q_error[tensor_indices] = q_before - q_after

    def run_episode(self, is_train=True, is_greedy=False):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        cumulative_reward = 0
        steps = 0
        episode_actions = []
        episode_states = []

        for step in range(self.max_steps):
            action = self.get_greedy_action(state) if is_greedy else self.choose_action(state)
            episode_states.append(state.copy())
            if isinstance(action, (int, float, np.number)):
                episode_actions.append(float(action))
            else:
                episode_actions.append(action.copy())
            state_prime, reward, done, info = self.env.step(action)

            if isinstance(info, dict) and 'terminal_observation' in info:
                done = done[0] if isinstance(done, tuple) else done

            # Convert reward to scalar if it's an array (for Pendulum compatibility)
            if isinstance(reward, np.ndarray):
                reward = float(reward.item()) if reward.size == 1 else float(reward)
            cumulative_reward += reward
            if len(state_prime.shape) > 1:
                state_prime = state_prime.flatten()

            if is_train:
                state_idx = self.discretizer.get_state_index(state)
                action_idx = self.discretizer.get_action_index(action)
                self.N[state_idx + action_idx] += 1
                self.N_total[state_idx] += 1
                self.update_q_matrix(state, action, state_prime, reward, done)

            if done:
                steps = step + 1
                break

            state = state_prime
            if (not is_greedy) & is_train & (self.epsilon > self.min_epsilon):
                self.epsilon *= self.decay
            if not is_greedy:
                self.alpha *= self.decay_alpha

        self.state_history.append(episode_states)
        self.action_history.append(episode_actions)
        steps = steps or self.max_steps
        
        if is_train and not is_greedy:
            self.current_episode += 1

        return steps, cumulative_reward

    def run_training_episode(self):
        n_steps, cumulative_reward = self.run_episode(is_train=True, is_greedy=False)
        self.training_steps.append(n_steps)
        self.training_cumulative_reward.append(cumulative_reward)
        return n_steps, cumulative_reward

    def run_greedy_episode(self):
        n_steps, cumulative_reward = self.run_episode(is_train=False, is_greedy=True)
        self.greedy_steps.append(n_steps)
        self.greedy_cumulative_reward.append(cumulative_reward)
        return n_steps, cumulative_reward

    def train(self, run_greedy_frequency=None, q_error_callback=None):
        self.training_cumulative_reward = []
        self.training_steps = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        
        print(f"\n{'='*60}")
        print(f"Training Started - Total Episodes: {self.episodes}")
        print(f"{'='*60}\n")
        
        if run_greedy_frequency:
            for episode in range(self.episodes):
                steps, episode_reward = self.run_episode(is_train=True, is_greedy=False)
                self.training_steps.append(steps)
                self.training_cumulative_reward.append(episode_reward)
                
                if episode > 0 and episode % max(1, self.episodes // 10) == 0:
                    recent_window = min(len(self.training_steps), 10)
                    avg_steps = np.mean(self.training_steps[-recent_window:])
                    avg_reward = np.mean(self.training_cumulative_reward[-recent_window:])
                    print(f"Episode {episode:4d} [Train] - Steps: {steps:4d}, Reward: {episode_reward:8.2f} | Avg(10): Steps={avg_steps:6.1f}, Reward={avg_reward:8.2f}")
                
                if (episode % run_greedy_frequency) == 0:
                    greedy_steps, greedy_reward = self.run_episode(is_train=False, is_greedy=True)
                    self.greedy_steps.append(greedy_steps)
                    self.greedy_cumulative_reward.append(greedy_reward)
                    print(f"Episode {episode:4d} [Greedy] - Steps: {greedy_steps:4d}, Reward: {greedy_reward:8.2f}")
                
                if q_error_callback and (episode < 2000 or episode % 100 == 0):
                    q_error_callback(self, episode)
        else:
            for episode in range(self.episodes):
                steps, episode_reward = self.run_episode(is_train=True, is_greedy=False)
                self.training_steps.append(steps)
                self.training_cumulative_reward.append(episode_reward)
                
                if episode > 0 and episode % max(1, self.episodes // 10) == 0:
                    recent_window = min(len(self.training_steps), 10)
                    avg_steps = np.mean(self.training_steps[-recent_window:])
                    avg_reward = np.mean(self.training_cumulative_reward[-recent_window:])
                    print(f"Episode {episode:4d} - Steps: {steps:4d}, Reward: {episode_reward:8.2f} | Avg(10): Steps={avg_steps:6.1f}, Reward={avg_reward:8.2f}")

    def evaluate_final_policy(self):
        rewards = []
        for _ in range(1000):
            _, cumulative_reward = self.run_episode(is_train=False, is_greedy=True)
            rewards.append(cumulative_reward)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)

    def measure_mean_runtime(self):
        state = self.env.reset()
        action = self.choose_action(state)
        state_prime, reward, done, _ = self.env.step(action)
        if len(state_prime.shape) > 1:
            state_prime = state_prime.flatten()
        start_time = time.time()
        for _ in range(100_000):
            self.update_q_matrix(state, action, state_prime, reward, done)
        end_time = time.time()
        return end_time - start_time

    def save_policy(self, filepath):
        def convert_to_list(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr
        policy_data = {
            'factors': [convert_to_list(factor) for factor in self.factors],
            'discretizer_params': {'dimensions': convert_to_list(self.discretizer.dimensions)},
            'Q_error': convert_to_list(self.Q_error),
            'N': convert_to_list(self.N),
            'N_total': convert_to_list(self.N_total)
        }
        np.savez(filepath, **policy_data)

    def convert_to_list(self, arr):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr

    def load_policy(self, filepath):
        policy_data = np.load(filepath, allow_pickle=True)
        self.factors = [np.array(factor) for factor in policy_data['factors']]
        self.factor_indices = np.arange(len(self.factors))
        self.non_factor_indices = [np.delete(self.factor_indices, i).tolist() for i in self.factor_indices]
        if 'Q_error' in policy_data:
            self.Q_error = np.array(policy_data['Q_error'])
        else:
            q_error_shape = tuple(self.discretizer.dimensions)
            self.Q_error = np.zeros(q_error_shape)
        if 'N' in policy_data:
            self.N = np.array(policy_data['N'])
        if 'N_total' in policy_data:
            self.N_total = np.array(policy_data['N_total'])

    def get_best_action(self, state):
        return self.get_greedy_action(state)

    def evaluate_policy(self, num_episodes=10):
        total_rewards = []
        for episode in range(num_episodes):
            _, episode_reward = self.run_episode(is_train=False, is_greedy=True)
            total_rewards.append(episode_reward)
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'rewards': total_rewards
        }

    def visualize_policy_behavior(self, num_steps=1000, render=False):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state_history = []
        action_history = []
        reward_history = []
        try:
            for step in range(num_steps):
                if render and hasattr(self.env, 'render'):
                    try:
                        self.env.render()
                    except Exception as e:
                        print(f"Warning: Render failed: {e}")
                        render = False
                state_history.append(np.copy(state))
                action = self.get_greedy_action(state)
                action_history.append(action)
                next_state, reward, done, info = self.env.step(action)
                if isinstance(info, dict) and 'terminal_observation' in info:
                    done = done[0] if isinstance(done, tuple) else done
                reward_history.append(reward)
                state = next_state
                if done:
                    break
        finally:
            if render and hasattr(self.env, 'close'):
                try:
                    self.env.close()
                except:
                    pass
        return {
            'states': np.array(state_history),
            'actions': np.array(action_history),
            'rewards': np.array(reward_history)
        }

    def plot_behavior(self, behavior_data):
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        time_steps = np.arange(len(behavior_data['rewards']))
        axes[0].plot(time_steps, behavior_data['states'][:, 0])
        axes[0].set_ylabel('Inventory Level')
        axes[0].grid(True)
        axes[1].plot(time_steps, behavior_data['states'][:, 1])
        axes[1].set_ylabel('Demand')
        axes[1].grid(True)
        axes[2].plot(time_steps, behavior_data['actions'])
        axes[2].set_ylabel('Action')
        axes[2].grid(True)
        axes[3].plot(time_steps, behavior_data['rewards'])
        axes[3].set_ylabel('Reward')
        axes[3].set_xlabel('Time Steps')
        axes[3].grid(True)
        plt.tight_layout()
        return fig