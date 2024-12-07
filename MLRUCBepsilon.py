from multiprocess import Pool

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.acrobot import CustomAcrobotEnv

from src.utils.utils import OOMFormatter
from src.algorithms.q_learning import QLearning
from src.utils.utils import Discretizer

import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class UCBMatrixQLearning:
    def __init__(
        self,
        env,
        discretizer,
        episodes=40000,
        max_steps=100,
        epsilon=0.4,
        alpha=0.1,  # 降低学习率
        gamma=0.95,
        decay=0.999999,
        min_epsilon=0.05,
        c=5.0,  # 降低UCB探索参数
        k=15,  # 增加矩阵秩
        warmstart_episodes=10000  # 预热阶段episodes数
    ):
        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.warmstart_episodes = warmstart_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.c = c
        self.k = k
        
        self.count_matrix = defaultdict(lambda: defaultdict(int))
        
        state_dims = tuple(self.discretizer.n_states)
        action_dims = tuple(self.discretizer.n_actions)
        self.L = (np.random.rand(*state_dims, k)) * 0.1  # 降低初始值
        self.R = (np.random.rand(k, *action_dims)) * 0.1
        
        self.rewards_per_episode = []
        self.steps_per_episode = []
    
    def get_q_value(self, state_idx, action_idx=None):
        if action_idx is None:
            return np.tensordot(self.L[state_idx], self.R, axes=([0], [0]))
        return np.tensordot(self.L[state_idx], self.R[:, action_idx], axes=([0], [0]))

    def select_action(self, state, current_episode):
        possible_actions = [tuple([i]) for i in range(self.discretizer.n_actions[0])]
        state_idx = self.discretizer.get_state_index(state)
        
        if current_episode < self.warmstart_episodes:
            # Warm start阶段: 仅使用epsilon-greedy
            if random.random() < self.epsilon:
                return random.choice(possible_actions)
        else:
            # UCB探索阶段
            if random.random() < self.epsilon:
                total_visits = sum(self.count_matrix[state_idx].values()) + 1
                ucb_values = []
                q_values = self.get_q_value(state_idx)
                
                for action in possible_actions:
                    action_idx = action[0]
                    if action in self.count_matrix[state_idx]:
                        count = self.count_matrix[state_idx][action]
                        ucb = q_values[action_idx] + self.c * np.sqrt(np.log(total_visits) / (count + 1))
                    else:
                        ucb = self.c * np.sqrt(np.log(total_visits))
                    ucb_values.append(ucb)
                
                return possible_actions[np.argmax(ucb_values)]
        
        # Greedy选择
        q_values = self.get_q_value(state_idx)
        return tuple([np.argmax(q_values)])
    
    def update_matrices(self, state_idx, action_idx, error_signal):
        L_s = self.L[state_idx]
        R_a = self.R[:, action_idx]
        
        grad_L = error_signal * R_a
        L_norm = np.linalg.norm(grad_L)
        if L_norm > 1e-8:
            self.L[state_idx] += self.alpha * grad_L / L_norm
            
        grad_R = error_signal * L_s
        R_norm = np.linalg.norm(grad_R)
        if R_norm > 1e-8:
            self.R[:, action_idx] += self.alpha * grad_R / R_norm
    
    def train(self, run_greedy_frequency=10):
        progress_bar = tqdm(range(self.episodes))
        
        for episode in progress_bar:
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state_idx = self.discretizer.get_state_index(state)
            total_reward = 0
            
            for step in range(self.max_steps):
                discretized_action = self.select_action(state, episode)
                action_idx = discretized_action[0]
                continuous_action = self.discretizer.get_action_from_index(discretized_action)

                next_state, reward, done, info = self.env.step(float(continuous_action[0]))
                if isinstance(info, dict) and 'terminal_observation' in info:
                    done = done[0] if isinstance(done, tuple) else done
                next_state_idx = self.discretizer.get_state_index(next_state)
                
                self.count_matrix[state_idx][discretized_action] += 1

                current_q = self.get_q_value(state_idx, action_idx)
                next_q = np.max(self.get_q_value(next_state_idx))
                error_signal = reward + self.gamma * next_q - current_q
                
                self.update_matrices(state_idx, action_idx, error_signal)
                
                total_reward += reward
                state = next_state
                state_idx = next_state_idx
                
                if done:
                    break
                    
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            
            self.rewards_per_episode.append(total_reward)
            self.steps_per_episode.append(step + 1)
            
            progress_bar.set_description(
                f"Episode {episode + 1}, "
                f"Reward: {total_reward:.2f}, "
                f"Steps: {step + 1}"
            )
            
            if (episode + 1) % run_greedy_frequency == 0:
                self.run_greedy()

    def run_greedy(self):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state_idx = self.discretizer.get_state_index(state)
        total_reward = 0
        
        for step in range(self.max_steps):
            q_values = self.get_q_value(state_idx)
            action_idx = np.argmax(q_values)
            continuous_action = self.discretizer.get_action_from_index(tuple([action_idx]))
            
            next_state, reward, done, _ = self.env.step(float(continuous_action[0]))
            state_idx = self.discretizer.get_state_index(next_state)
            
            total_reward += reward
            
            if done:
                break
        
        return total_reward, step + 1

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(self.rewards_per_episode)
        ax1.axvline(x=self.warmstart_episodes, color='r', linestyle='--', label='Warm Start End')
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        
        ax2.plot(self.steps_per_episode)
        ax2.axvline(x=self.warmstart_episodes, color='r', linestyle='--', label='Warm Start End')
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

env = CustomContinuousCartPoleEnv()
discretizer = Discretizer(
    min_points_states=[-4.8, -0.5, -0.42, -0.9],
    max_points_states=[4.8, 0.5, 0.42, 0.9],
    bucket_states=[10, 10, 10, 10],
    min_points_actions=[-1],
    max_points_actions=[1],
    bucket_actions=[20]
)

learner = UCBMatrixQLearning(
    env=env,
    discretizer=discretizer,
    episodes=40000,
    max_steps=100,
    epsilon=0.4,
    alpha=0.01,
    gamma=0.9,
    decay=0.99999999,
    min_epsilon=0.05,
    c=5.0,
    k=4,
    warmstart_episodes=10000
)

learner.train(run_greedy_frequency=10)
learner.plot_results()