import numpy as np
import time
import tensorly as tl
import torch
from collections import deque
import random
import matplotlib.pyplot as plt
from src.environments.wireless import WirelessCommunicationsEnv

class ReplayBuffer:
    """经验回放缓冲区，高效存储状态-动作对的访问次数和Q_error"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        # 使用字典快速查找状态-动作访问计数
        self.state_action_counts = {}
        # 使用字典快速查找状态访问计数
        self.state_counts = {}
        # 使用字典快速查找Q误差
        self.q_errors = {}

    def add(self, state_idx, action_idx, q_error):
        """添加经验到缓冲区并更新计数器"""
        # 创建键值
        state_key = tuple(state_idx)
        state_action_key = state_key + tuple(action_idx) 
        
        # 更新访问计数
        if state_key in self.state_counts:
            self.state_counts[state_key] += 1
        else:
            self.state_counts[state_key] = 1
            
        if state_action_key in self.state_action_counts:
            self.state_action_counts[state_action_key] += 1
        else:
            self.state_action_counts[state_action_key] = 1
        
        # 更新Q_error
        self.q_errors[state_action_key] = q_error
        
        # 添加到缓冲区
        self.buffer.append((state_idx, action_idx, q_error))

    def sample(self, batch_size=1):
        """从缓冲区随机采样经验"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def get_state_count(self, state_idx):
        """获取状态的访问次数"""
        state_key = tuple(state_idx)
        return self.state_counts.get(state_key, 0)
    
    def get_state_action_count(self, state_idx, action_idx):
        """获取状态-动作对的访问次数"""
        state_action_key = tuple(state_idx) + tuple(action_idx)
        return self.state_action_counts.get(state_action_key, 0)
    
    def get_q_error(self, state_idx, action_idx):
        """获取状态-动作对的Q_error"""
        state_action_key = tuple(state_idx) + tuple(action_idx)
        return self.q_errors.get(state_action_key, 0)
    
    def __len__(self):
        return len(self.buffer)


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
        convergence_threshold=0.01,
        max_inner_iterations=100,
        ucb_c=1.0,
        lambda_penalty=1.0,
        epsilon_penalty=1e-6,
        buffer_size=10000,  # 新增参数
    ):
        # 原有初始化代码
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

        self.factors = [(np.random.rand(dim, self.k) - bias) * init_ord for dim in self.discretizer.dimensions]
        self.factor_indices = np.arange(len(self.factors))
        self.non_factor_indices = [np.delete(self.factor_indices, i).tolist() for i in self.factor_indices]

        # 保持接口一致，仍然创建这些属性
        q_error_shape = tuple(self.discretizer.dimensions)
        self.Q_error = np.zeros(q_error_shape)
        self.N = np.zeros(q_error_shape, dtype=np.float64)
        self.N_total = np.zeros(tuple(self.discretizer.n_states), dtype=np.float64)

        # 添加经验回放区
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.action_history = []
        self.state_history = []

    # 更新Q矩阵方法，保持相同接口但内部使用回放区
    def update_q_matrix(self, state, action, state_prime, reward, done):
        state_idx = self.discretizer.get_state_index(state)
        state_prime_idx = self.discretizer.get_state_index(state_prime)
        action_idx = self.discretizer.get_action_index(action)

        # 计算目标Q值
        q_next = np.max(self.get_q_from_state_idx(state_prime_idx)) if not done else 0
        target_q = reward + self.gamma * q_next

        # 更新前的Q值
        q_before = self.get_q_from_state_action_idx(state_idx, action_idx)
        tensor_indices = state_idx + action_idx
        
        # 从回放区获取访问计数（没有则为0）
        sa_key = tuple(tensor_indices)
        n_sa = self.replay_buffer.get_state_action_count(state_idx, action_idx)
        
        # 更新访问计数矩阵，仅作为兼容性
        self.N[tensor_indices] = n_sa + 1
        self.N_total[state_idx] += 1
        
        # 以下是原有的优化方法，保持不变
        iteration = 0
        q_prev = q_before
        penalty_weight = 1.0 / (n_sa + self.epsilon_penalty)

        while iteration < self.max_inner_iterations:
            # 原有优化代码
            new_factors = [factor.copy() for factor in self.factors]
            for factor_idx in self.factor_indices:
                grad_factor = np.ones(self.k)
                for non_factor_idx in self.non_factor_indices[factor_idx]:
                    grad_factor *= self.factors[non_factor_idx][tensor_indices[non_factor_idx], :]

                q_current = self.get_q_from_state_action_idx(state_idx, action_idx)
                error_signal = target_q - q_current
                error_update = -error_signal * grad_factor / np.linalg.norm(grad_factor)

                penalty_direction = np.sign(q_current)
                penalty_update = penalty_weight * penalty_direction * grad_factor / np.linalg.norm(grad_factor)

                update = error_update + self.lambda_penalty * penalty_update
                new_factors[factor_idx][tensor_indices[factor_idx], :] -= self.alpha * update

            self.factors = new_factors
            if self.normalize_columns:
                self.normalize()

            q_new = self.get_q_from_state_action_idx(state_idx, action_idx)
            if abs(q_new - q_prev) < self.convergence_threshold:
                break
            q_prev = q_new
            iteration += 1

        print(f"Episode {self.current_episode}, Iterations to converge: {iteration}")

        # 计算Q_error
        q_after = self.get_q_from_state_action_idx(state_idx, action_idx)
        q_error = abs(q_before - q_after)
        
        # 更新Q_error矩阵以保持接口一致
        self.Q_error[tensor_indices] = q_error
        
        # 同时添加到经验回放区
        self.replay_buffer.add(state_idx, action_idx, q_error)

    def run_episode(self, is_train=True, is_greedy=False):
    
        # 保持原有代码
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

            cumulative_reward += reward
            if len(state_prime.shape) > 1:
                state_prime = state_prime.flatten()

            if is_train:
                # 不直接更新N和N_total，改由update_q_matrix管理
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
    
    def get_random_action(self):
        if isinstance(self.env, WirelessCommunicationsEnv):
            K = self.env.K
            return np.random.uniform(0, 2, size=K)
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
        # Epsilon-Greedy: 以 epsilon 概率选择随机动作
        if np.random.random() < self.epsilon:
            return self.get_random_action()

        # UCB 策略
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
        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.current_episode = 0
        
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
            'Q_error': convert_to_list(self.Q_error),
            'N': convert_to_list(self.N),
            'N_total': convert_to_list(self.N_total)
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
        if 'N' in policy_data:
            self.N = np.array(policy_data['N'])
        if 'N_total' in policy_data:
            self.N_total = np.array(policy_data['N_total'])

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



