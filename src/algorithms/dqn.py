"""
DQN with Paper's True Architecture (Compressed)
================================================
Implements DQN exactly as in the paper's code with minimal parameters.

Key Architecture:
- Input: Bucket indices (NOT flat index, NO embedding)
- CartPole: [bucket_x, bucket_x_dot, bucket_theta, bucket_theta_dot] = 4 integers
- MLP processes these integers directly
- Output: Q-values for all actions

Parameter Count (CartPole, arch=[64]):
- Input: 4 dimensions
- Hidden: 64 neurons
- Output: 10 actions
- Total: 4×64 + 64 + 64×10 + 10 = 970 parameters

This achieves ~1000 params comparable to tensor methods!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Limit torch threads for fair comparison
torch.set_num_threads(1)


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state_buckets, action_idx, reward, next_state_buckets, done):
        self.buffer.append((state_buckets, action_idx, reward, next_state_buckets, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class SimpleMLP(nn.Module):
    """
    Simple MLP that takes bucket indices as input.
    
    Architecture:
        bucket_indices[n_dims] → FC1 → ReLU → ... → FCn → Q-values[n_actions]
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, bucket_indices):
        x = bucket_indices.float()
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class DQN:
    """
    DQN using paper's true architecture with compressed parameters.
    """
    
    def __init__(
        self,
        env,
        discretizer,
        episodes,
        max_steps,
        arch=[64],
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999999,
        min_epsilon=0.01,
        batch_size=32,
        buffer_size=100000,
        warmup_steps=500,
        update_freq=1,
        target_update_freq=100,
        # Compatibility parameters (unused)
        alpha=None, decay=None, embed_dim=None, hidden_dim=None,
        tau=None, k=None, bias=None, **kwargs
    ):
        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.arch = arch
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        
        self.n_state_dims = len(discretizer.n_states)
        self.n_actions = int(np.prod(discretizer.n_actions))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Q-networks
        self.q_network = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        self.target_network = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Result tracking (compatible with TEQL interface)
        self.training_steps = []
        self.training_cumulative_reward = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        self.action_history = []
        self.state_history = []
        
        self.total_steps = 0
        self.losses = []
        
        self._print_hyperparameters()
    
    def _print_hyperparameters(self):
        """Print all hyperparameter configurations - matching TEQL style."""
        total_params = self.q_network.count_parameters()
        
        print("\n" + "="*70)
        print("DQN Hyperparameters Configuration")
        print("="*70)
        print(f"Environment Parameters:")
        print(f"  episodes:             {self.episodes}")
        print(f"  max_steps:            {self.max_steps}")
        print(f"\nRL Core Parameters:")
        print(f"  epsilon (initial):    {self.epsilon}")
        print(f"  min_epsilon:          {self.min_epsilon}")
        print(f"  learning_rate:        {self.optimizer.param_groups[0]['lr']}")
        print(f"  gamma (discount):     {self.gamma}")
        print(f"  epsilon_decay:        {self.epsilon_decay}")
        print(f"\nNetwork Architecture:")
        print(f"  input_dim:            {self.n_state_dims} (bucket indices)")
        print(f"  hidden_layers:        {self.arch}")
        print(f"  output_dim:           {self.n_actions}")
        print(f"\nReplay Buffer:")
        print(f"  batch_size:           {self.batch_size}")
        print(f"  buffer_size:          {self.replay_buffer.buffer.maxlen:,}")
        print(f"  warmup_steps:         {self.warmup_steps}")
        print(f"\nDiscretization:")
        print(f"  state dimensions:     {list(self.discretizer.n_states)}")
        print(f"  action dimensions:    {list(self.discretizer.n_actions)}")
        print(f"\nParameter Count:        {total_params:,}")
        print("="*70 + "\n")
    
    def get_state_buckets(self, state):
        return self.discretizer.get_state_index(state)
    
    def get_action_value(self, action_idx):
        """Convert action index to actual action value."""
        # Safety check: ensure action_idx is within valid range
        if action_idx < 0 or action_idx >= self.n_actions:
            print(f"Warning: action_idx {action_idx} out of range [0, {self.n_actions}), clipping")
            action_idx = max(0, min(action_idx, self.n_actions - 1))
        
        if self.discretizer.discrete_action:
            # For discrete action spaces (like highway), return integer action directly
            return int(action_idx)
        else:
            action_indices = np.unravel_index(action_idx, tuple(self.discretizer.n_actions))
            action_value = self.discretizer.get_action_from_index(action_indices)
            
            # For single-dimensional continuous action (like Pendulum), return scalar
            if isinstance(action_value, np.ndarray):
                if action_value.size == 1:
                    return float(action_value.item())
                action_value = np.squeeze(action_value)
            return action_value
    
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            state_buckets = self.get_state_buckets(state)
            
            # Ensure state_buckets is a flat list/tuple of integers
            if isinstance(state_buckets, tuple):
                state_buckets = list(state_buckets)
            
            # Convert to tensor with correct shape [1, n_state_dims]
            state_tensor = torch.FloatTensor(state_buckets).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            # q_values shape should be [1, n_actions], get the best action
            return q_values[0].argmax().item()
    
    def update(self):
        if len(self.replay_buffer) < self.warmup_steps:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def train(self, run_greedy_frequency=None, q_error_callback=None):
        """Train DQN agent - matching TEQL interface."""
        self.training_cumulative_reward = []
        self.training_steps = []
        self.greedy_steps = []
        self.greedy_cumulative_reward = []
        
        print(f"\n{'='*60}")
        print(f"Training Started - Total Episodes: {self.episodes}")
        print(f"{'='*60}\n")
        
        for episode in range(self.episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            # Ensure state is 1D
            if hasattr(state, 'shape') and len(state.shape) > 1:
                state = state.flatten()
            episode_reward = 0
            episode_steps = 0
            episode_actions = []
            episode_states = []
            
            for step in range(self.max_steps):
                action_idx = self.select_action(state, self.epsilon)
                action_value = self.get_action_value(action_idx)
                
                episode_states.append(state.copy() if hasattr(state, 'copy') else state)
                if isinstance(action_value, np.ndarray):
                    episode_actions.append(action_value.copy())
                else:
                    episode_actions.append(float(action_value))
                
                next_state, reward, done, info = self.env.step(action_value)
                
                # Ensure next_state is 1D
                if hasattr(next_state, 'shape') and len(next_state.shape) > 1:
                    next_state = next_state.flatten()
                
                if isinstance(info, dict) and 'terminal_observation' in info:
                    done = done[0] if isinstance(done, tuple) else done
                
                # Convert reward to scalar if needed
                if isinstance(reward, np.ndarray):
                    reward = float(reward.item()) if reward.size == 1 else float(reward)
                
                state_buckets = self.get_state_buckets(state)
                next_state_buckets = self.get_state_buckets(next_state)
                self.replay_buffer.push(state_buckets, action_idx, reward, next_state_buckets, done)
                
                if self.total_steps % self.update_freq == 0:
                    loss = self.update()
                    if loss is not None:
                        self.losses.append(loss)
                
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                state = next_state
                
                if done:
                    break
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            self.training_steps.append(episode_steps)
            self.training_cumulative_reward.append(episode_reward)
            self.state_history.append(episode_states)
            self.action_history.append(episode_actions)
            
            # Print progress - matching TEQL style
            if episode > 0 and episode % max(1, self.episodes // 10) == 0:
                recent_window = min(len(self.training_steps), 10)
                avg_steps = np.mean(self.training_steps[-recent_window:])
                avg_reward = np.mean(self.training_cumulative_reward[-recent_window:])
                print(f"Episode {episode:4d} [Train] - Steps: {episode_steps:4d}, Reward: {episode_reward:8.2f} | Avg(10): Steps={avg_steps:6.1f}, Reward={avg_reward:8.2f}")
            
            # Greedy evaluation - matching TEQL style
            if run_greedy_frequency and (episode % run_greedy_frequency) == 0:
                greedy_steps, greedy_reward = self.run_greedy_episode()
                print(f"Episode {episode:4d} [Greedy] - Steps: {greedy_steps:4d}, Reward: {float(greedy_reward):8.2f}")
        
        return self
    
    def run_greedy_episode(self):
        """Run a single greedy episode."""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        # Ensure state is 1D
        if hasattr(state, 'shape') and len(state.shape) > 1:
            state = state.flatten()
        episode_reward = 0
        steps = 0
        
        for step in range(self.max_steps):
            action_idx = self.select_action(state, epsilon=0.0)
            action_value = self.get_action_value(action_idx)
            next_state, reward, done, info = self.env.step(action_value)
            
            # Ensure next_state is 1D
            if hasattr(next_state, 'shape') and len(next_state.shape) > 1:
                next_state = next_state.flatten()
            
            if isinstance(info, dict) and 'terminal_observation' in info:
                done = done[0] if isinstance(done, tuple) else done
            
            if isinstance(reward, np.ndarray):
                reward = float(reward.item()) if reward.size == 1 else float(reward)
            
            episode_reward += reward
            state = next_state
            steps = step + 1
            
            if done:
                break
        
        self.greedy_steps.append(steps)
        self.greedy_cumulative_reward.append(episode_reward)
        
        return steps, episode_reward
    
    def evaluate_final_policy(self):
        """Evaluate final policy - matching TEQL interface."""
        rewards = []
        for _ in range(1000):
            _, cumulative_reward = self.run_greedy_episode()
            # Pop the last entry since run_greedy_episode adds to lists
            self.greedy_steps.pop()
            self.greedy_cumulative_reward.pop()
            rewards.append(cumulative_reward)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)