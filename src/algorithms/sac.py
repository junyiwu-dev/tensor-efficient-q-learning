"""
SAC with Paper's True Architecture (Compressed)
================================================
≈dre4wsaq   
Implements SAC using the same architecture principle as paper's DQN.

Key Architecture:
- Input: Bucket indices (NOT flat index, NO embedding)
- Actor: bucket_indices → MLP → action probabilities
- Critic: bucket_indices → MLP → Q-values
- Twin critics for stability

Parameter Count Example (CartPole, arch=[20]):
- Each network: 4×20 + 20 + 20×10 + 10 = 310
- Actor: 310
- Critic1: 310
- Critic2: 310
- Temperature: 1
- Total: 931 parameters

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
    """Simple MLP for both Actor and Critic."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
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


class SAC:
    """
    Soft Actor-Critic using paper's architecture with compressed parameters.
    """
    
    def __init__(
        self,
        env,
        discretizer,
        episodes,
        max_steps,
        arch=[20],
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        init_temperature=0.2,
        learnable_temperature=True,
        batch_size=64,
        buffer_size=100000,
        warmup_steps=500,
        update_freq=1,
        # Compatibility parameters (unused)
        lr=None, epsilon=None, alpha=None, decay=None,
        embed_dim=None, hidden_dim=None, k=None, bias=None, **kwargs
    ):
        self.env = env
        self.discretizer = discretizer
        self.episodes = episodes
        self.max_steps = max_steps
        self.arch = arch
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        
        self.n_state_dims = len(discretizer.n_states)
        self.n_actions = int(np.prod(discretizer.n_actions))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Actor network
        self.actor = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        # Create twin Critic networks
        self.critic1 = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        self.critic2 = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        # Create target critics
        self.critic1_target = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        self.critic2_target = SimpleMLP(
            input_dim=self.n_state_dims,
            hidden_dims=arch,
            output_dim=self.n_actions
        ).to(self.device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Temperature parameter
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            self.log_alpha = torch.tensor(np.log(init_temperature), 
                                         requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.target_entropy = -np.log(1.0 / self.n_actions) * 0.98
        else:
            self._alpha = init_temperature
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Result tracking - matching TEQL interface
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
        actor_params = self.actor.count_parameters()
        critic_params = self.critic1.count_parameters()
        temp_params = 1 if self.learnable_temperature else 0
        total_params = actor_params + 2 * critic_params + temp_params
        
        print("\n" + "="*70)
        print("SAC Hyperparameters Configuration")
        print("="*70)
        print(f"Environment Parameters:")
        print(f"  episodes:             {self.episodes}")
        print(f"  max_steps:            {self.max_steps}")
        print(f"\nRL Core Parameters:")
        print(f"  actor_lr:             {self.actor_optimizer.param_groups[0]['lr']}")
        print(f"  critic_lr:            {self.critic1_optimizer.param_groups[0]['lr']}")
        print(f"  gamma (discount):     {self.gamma}")
        print(f"  tau (soft update):    {self.tau}")
        print(f"\nNetwork Architecture:")
        print(f"  input_dim:            {self.n_state_dims} (bucket indices)")
        print(f"  hidden_layers:        {self.arch}")
        print(f"  output_dim:           {self.n_actions}")
        print(f"\nTemperature:")
        if self.learnable_temperature:
            print(f"  learnable:            True")
            print(f"  initial α:            {torch.exp(self.log_alpha).item():.3f}")
            print(f"  target_entropy:       {self.target_entropy:.3f}")
        else:
            print(f"  learnable:            False")
            print(f"  α:                    {self._alpha:.3f}")
        print(f"\nReplay Buffer:")
        print(f"  batch_size:           {self.batch_size}")
        print(f"  buffer_size:          {self.replay_buffer.buffer.maxlen:,}")
        print(f"  warmup_steps:         {self.warmup_steps}")
        print(f"\nDiscretization:")
        print(f"  state dimensions:     {list(self.discretizer.n_states)}")
        print(f"  action dimensions:    {list(self.discretizer.n_actions)}")
        print(f"\nParameter Count:")
        print(f"  Actor:                {actor_params:,}")
        print(f"  Critic (×2):          {2 * critic_params:,}")
        if temp_params:
            print(f"  Temperature:          {temp_params}")
        print(f"  Total:                {total_params:,}")
        print("="*70 + "\n")
    
    @property
    def alpha(self):
        if self.learnable_temperature:
            return torch.exp(self.log_alpha).item()
        else:
            return self._alpha
    
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
    
    def select_action(self, state, deterministic=False):
        state_buckets = self.get_state_buckets(state)
        
        # Ensure state_buckets is a flat list/tuple of integers
        if isinstance(state_buckets, tuple):
            state_buckets = list(state_buckets)
        
        # Convert to tensor with correct shape [1, n_state_dims]
        state_tensor = torch.FloatTensor(state_buckets).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action_idx = probs[0].argmax().item()
            else:
                dist = torch.distributions.Categorical(probs[0])
                action_idx = dist.sample().item()
        
        return action_idx
    
    def update(self):
        if len(self.replay_buffer) < self.warmup_steps:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_q1 = self.critic1_target(next_states)
            next_q2 = self.critic2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=1)
            target_q = rewards + (1 - dones) * self.gamma * next_v
        
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)
        
        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        if self.learnable_temperature:
            with torch.no_grad():
                entropy = -(probs * log_probs).sum(dim=1).mean()
            
            alpha_loss = -self.log_alpha * (entropy - self.target_entropy)
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        for target, source in [(self.critic1_target, self.critic1),
                               (self.critic2_target, self.critic2)]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        return (critic1_loss.item() + critic2_loss.item()) / 2
    
    def train(self, run_greedy_frequency=None, q_error_callback=None):
        """Train SAC agent - matching TEQL interface."""
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
                action_idx = self.select_action(state, deterministic=False)
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
                
                if isinstance(reward, np.ndarray):
                    reward = float(reward.item()) if reward.size == 1 else float(reward)
                
                state_buckets = self.get_state_buckets(state)
                next_state_buckets = self.get_state_buckets(next_state)
                self.replay_buffer.push(state_buckets, action_idx, reward, 
                                       next_state_buckets, done)
                
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
        """Run a single greedy (deterministic) episode."""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        # Ensure state is 1D
        if hasattr(state, 'shape') and len(state.shape) > 1:
            state = state.flatten()
        episode_reward = 0
        steps = 0
        
        for step in range(self.max_steps):
            action_idx = self.select_action(state, deterministic=True)
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
            self.greedy_steps.pop()
            self.greedy_cumulative_reward.pop()
            rewards.append(cumulative_reward)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)