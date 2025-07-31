import numpy as np
import time
import tensorly as tl
import torch
import matplotlib.pyplot as plt

from src.environments.wireless import WirelessCommunicationsEnv
from src.utils.utils import DiscretizerTorch, ReplayBufferTorch
from src.models.mlp import PARAFAC

torch.set_num_threads(1)

class TensorLowRankLearning:
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

        self.factors = [(np.random.rand(dim, self.k) - bias) * init_ord for dim in self.discretizer.dimensions]
        self.factor_indices = np.arange(len(self.factors))
        self.non_factor_indices = [np.delete(self.factor_indices, i).tolist() for i in self.factor_indices]

        q_error_shape = tuple(self.discretizer.dimensions)
        self.Q_error = np.zeros(q_error_shape)

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.action_history = []
        self.state_history = []

    def get_random_action(self):
        # 返回连续动作，兼容 Acrobot 和其他环境
        if isinstance(self.env, WirelessCommunicationsEnv):
            K = self.env.K
            return np.random.uniform(0, 2, size=K)
        # 对于 Acrobot 和其他连续动作环境，返回标量或向量
        return np.random.uniform(self.discretizer.min_points_actions, self.discretizer.max_points_actions)

    def get_greedy_action(self, state):
        state_idx = self.discretizer.get_state_index(state)
        q_simplified = self.get_q_from_state_idx(state_idx).reshape(self.discretizer.n_actions)
        action_idx = np.unravel_index(np.argmax(q_simplified), q_simplified.shape)
        return self.discretizer.get_action_from_index(action_idx)

    def get_q_from_state_idx(self, state_idx):
        state_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors[:len(state_idx)]):
            factor_row = factor[state_idx[idx], :]
            state_value *= factor_row
        action_khatri = tl.tenalg.khatri_rao(self.factors[len(state_idx):])
        action_state = np.dot(state_value, action_khatri.T)
        return action_state

    def get_q_from_state_action_idx(self, state_idx, action_idx):
        indices = state_idx + action_idx
        q_value = np.ones(self.k)
        for idx, factor in enumerate(self.factors):
            q_value *= factor[indices[idx], :]
        return np.sum(q_value)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = self.get_random_action()
        else:
            action = self.get_greedy_action(state)
        return action

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
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        q_next = np.max(self.get_q_from_state_idx(state_prime_idx)) if not done else 0
        target_q = reward + self.gamma * q_next

        q_before = self.get_q_from_state_action_idx(state_idx, action_idx)

        error_signal = target_q - q_before
        tensor_indices = state_idx + action_idx

        new_factors = [factor.copy() for factor in self.factors]
        for factor_idx in self.factor_indices:
            grad_factor = np.ones(self.k)
            for non_factor_idx in self.non_factor_indices[factor_idx]:
                grad_factor *= self.factors[non_factor_idx][tensor_indices[non_factor_idx], :]
            update = -error_signal * grad_factor / np.linalg.norm(grad_factor)
            new_factors[factor_idx][tensor_indices[factor_idx], :] -= self.alpha * update
        
        self.factors = new_factors

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
            # 动态处理动作类型，兼容标量和数组
            if isinstance(action, (int, float, np.number)):
                episode_actions.append(float(action))  # 标量动作转为浮点数
            else:
                episode_actions.append(action.copy())  # 数组动作使用 copy()
            state_prime, reward, done, info = self.env.step(action)

            if isinstance(info, dict) and 'terminal_observation' in info:
                done = done[0] if isinstance(done, tuple) else done

            cumulative_reward += reward
            if len(state_prime.shape) > 1:
                state_prime = state_prime.flatten()

            if is_train:
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
        
        if run_greedy_frequency:
            for episode in range(self.episodes):
                steps, episode_reward = self.run_episode(is_train=True, is_greedy=False)
                self.training_steps.append(steps)
                self.training_cumulative_reward.append(episode_reward)
                if (episode % run_greedy_frequency) == 0:
                    greedy_steps, greedy_reward = self.run_episode(is_train=False, is_greedy=True)
                    self.greedy_steps.append(greedy_steps)
                    self.greedy_cumulative_reward.append(greedy_reward)
                if q_error_callback and (episode < 2000 or episode % 100 == 0):
                    q_error_callback(self, episode)
        else:
            for episode in range(self.episodes):
                steps, episode_reward = self.run_episode(is_train=True, is_greedy=False)
                self.training_steps.append(steps)
                self.training_cumulative_reward.append(episode_reward)
                if q_error_callback and (episode < 2000 or episode % 100 == 0):
                    q_error_callback(self, episode)

        self.evaluate_final_policy()

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
            'Q_error': convert_to_list(self.Q_error)
        }
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        import json
        with open(filepath, 'w') as f:
            json.dump(policy_data, f)

    def load_policy(self, filepath):
        import json
        import numpy as np
        with open(filepath, 'r') as f:
            policy_data = json.load(f)
        self.factors = [np.array(factor) for factor in policy_data['factors']]
        if 'Q_error' in policy_data:
            self.Q_error = np.array(policy_data['Q_error'])
        else:
            q_error_shape = tuple(self.discretizer.dimensions)
            self.Q_error = np.zeros(q_error_shape)

    def get_best_action(self, state):
        return self.get_greedy_action(state)

    def evaluate_policy(self, num_episodes=10):
        total_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            episode_reward = 0
            for step in range(self.max_steps):
                action = self.get_greedy_action(state)
                next_state, reward, done, info = self.env.step(action)
                if isinstance(info, dict) and 'terminal_observation' in info:
                    done = done[0] if isinstance(done, tuple) else done
                episode_reward += reward
                state = next_state
                if done:
                    break
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
        states = np.array(state_history)
        actions = np.array(action_history)
        rewards = np.array(reward_history)
        if states.shape[1] == 2:
            angles = states[:, 0]
            angular_velocities = states[:, 1]
        elif states.shape[1] == 3:
            angles = np.arctan2(states[:, 1], states[:, 0])
            angular_velocities = states[:, 2]
        else:
            raise ValueError(f"Unexpected state shape: {states.shape}")
        return {
            'angles': angles,
            'angular_velocities': angular_velocities,
            'actions': actions,
            'rewards': rewards,
            'states': states
        }

    def plot_behavior(self, behavior_data):
        time_steps = np.arange(len(behavior_data['angles']))
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle('Pendulum Behavior Analysis')
        axes[0].plot(time_steps, behavior_data['angles'] * 180 / np.pi)
        axes[0].set_ylabel('Angle (degrees)')
        axes[0].grid(True)
        axes[1].plot(time_steps, behavior_data['angular_velocities'])
        axes[1].set_ylabel('Angular Velocity')
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

class TensorLowRankLearningTorch:
    def __init__(
        self,
        discretizer: DiscretizerTorch,
        alpha: float,
        gamma: float,
        k: int,
        scale: float,
        bias: float,
        method: int,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.method = method
        self.buffer = ReplayBufferTorch(1)
        self.discretizer = discretizer
        self.Q = PARAFAC(
            np.concatenate([discretizer.bucket_states, discretizer.bucket_actions]),
            k=k,
            scale=scale,
            nA=len(discretizer.bucket_actions),
            bias=bias,
        ).double()
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=alpha)

    def select_random_action(self) -> np.ndarray:
        a_idx = tuple(np.random.randint(self.discretizer.bucket_actions).tolist())
        return self.discretizer.get_action_from_index(a_idx)

    def select_greedy_action(self, s: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s_idx = np.concatenate([self.discretizer.get_state_index(s)])
            a_idx_flat = self.Q(s_idx).argmax().detach().item()
            a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_actions)
            return self.discretizer.get_action_from_index(a_idx)

    def select_action(self, s: np.ndarray, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            return self.select_random_action()
        return self.select_greedy_action(s)

    def update(self) -> None:
        s, a, sp, r, d = self.buffer.sample()
        s_idx = np.concatenate([self.discretizer.get_state_index(s)])
        sp_idx = np.concatenate([self.discretizer.get_state_index(sp)])
        a_idx = self.discretizer.get_action_index(a)
        q_target = r + self.gamma * self.Q(sp_idx).max().detach()
        q_hat = self.Q(np.concatenate([s_idx, a_idx]))
        if self.method == 0:
            self.opt.zero_grad()
            loss = torch.nn.MSELoss()
            loss(q_hat, q_target).backward()
            self.opt.step()
        elif self.method == 1:
            for factor in self.Q.factors:
                q_target = r + self.gamma * self.Q(sp_idx).max().detach()
                q_hat = self.Q(np.concatenate([s_idx, a_idx]))
                self.opt.zero_grad()
                loss = torch.nn.MSELoss()
                loss(q_hat, q_target).backward()
                with torch.no_grad():
                    for frozen_factor in self.Q.factors:
                        if frozen_factor is not factor:
                            frozen_factor.grad = None
                self.opt.step()