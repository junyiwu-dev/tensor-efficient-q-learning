"""
Unified Experiment Runner
==========================
Supports all algorithms: TLR, TEQL, LoRa-VI, DQN, SAC (Discrete)

All algorithms share:
- Same Discretizer for state/action discretization
- Same interface (train, run_episode, etc.)
- Same result format for fair comparison
"""

import os
import sys
import json
import pickle
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithms'))

from src.algorithms.tlr_original import TensorLowRankLearning as OriginalTLR
from src.algorithms.teql import TensorEfficientQL as TEQL
from src.algorithms.lora_vi import LoRaVI
from src.algorithms.dqn import DQN
from src.algorithms.sac import SAC as DiscreteSAC
from utils import Discretizer


class Experiment:
    """
    Unified experiment runner for all algorithms.
    
    Supports:
    - original-tlr / tlr: Original Tensor Low-Rank Learning
    - convergent-tlr / teql: Tensor-Efficient Q-Learning
    - lora-vi: Low-Rank Value Iteration
    - dqn: Deep Q-Network (Tabular)
    - sac: Soft Actor-Critic (Discrete)
    """
    
    def __init__(self, name, env, recover=False, run_freq=10, param_dir='parameters'):
        self.env = env
        self.name = name
        self.run_freq = run_freq
        self.param_dir = param_dir

        with open(os.path.join(param_dir, name), 'r') as f:
            self.parameters = json.load(f)

        # All methods need discretizer for consistent action space
        self.discretizer = self._get_discretizer()

        if recover:
            self.models = self._get_models_from_checkpoints()
        else:
            self.models = self._get_models()

    def _get_discretizer(self):
        """Create discretizer for all methods."""
        states_structure = self.parameters.get('states_structure', None)
        discrete_action = self.parameters.get('discrete_action', False)
        
        bucket_states = np.array(self.parameters['bucket_states'], dtype=np.int32)
        bucket_actions = np.array(self.parameters['bucket_actions'], dtype=np.int32)
        
        discretizer = Discretizer(
            min_points_states=self.parameters['min_points_states'],
            max_points_states=self.parameters['max_points_states'],
            bucket_states=bucket_states,
            min_points_actions=self.parameters['min_points_actions'],
            max_points_actions=self.parameters['max_points_actions'],
            bucket_actions=bucket_actions,
            states_structure=states_structure,
            discrete_action=discrete_action
        )
        
        discretizer.n_states = np.array(discretizer.n_states, dtype=np.int32)
        discretizer.n_actions = np.array(discretizer.n_actions, dtype=np.int32)
        discretizer.dimensions = np.concatenate([discretizer.n_states, discretizer.n_actions]).astype(np.int32)
        
        return discretizer

    def _get_models(self):
        model_type = self.parameters.get('type', 'original-tlr')
        
        # Normalize model type names
        model_type = model_type.lower().replace('_', '-')
        
        if model_type in ['original-tlr', 'tlr']:
            return self._get_original_tlr_models()
        elif model_type in ['convergent-tlr', 'teql']:
            return self._get_teql_models()
        elif model_type in ['lora-vi']:
            return self._get_lora_vi_models()
        elif model_type in ['dqn']:
            return self._get_dqn_models()
        elif model_type in ['sac']:
            return self._get_sac_discrete_models()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _get_original_tlr_models(self):
        """Create Original TLR model."""
        bias = self.parameters.get('bias', 0.0)
        return [OriginalTLR(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=self.parameters['k'],
            bias=bias
        )]

    def _get_teql_models(self):
        """Create TEQL model."""
        bias = self.parameters.get('bias', 0.0)
        return [TEQL(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters['decay'],
            k=self.parameters['k'],
            bias=bias,
            convergence_threshold=self.parameters.get('convergence_threshold', 1e-4),
            max_inner_iterations=self.parameters.get('max_inner_iterations', 100),
            ucb_c=self.parameters.get('ucb_c', 1.0),
            lambda_penalty=self.parameters.get('lambda_penalty', 0),
            epsilon_penalty=self.parameters.get('epsilon_penalty', 1e-4)
        )]

    def _get_lora_vi_models(self):
        """Create LoRa-VI model."""
        bias = self.parameters.get('bias', 0.0)
        return [LoRaVI(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters.get('decay', 1.0),
            k=self.parameters['k'],
            bias=bias,
            max_inner_iterations=self.parameters.get('max_inner_iterations', 5),
            policy_improvement_freq=self.parameters.get('policy_improvement_freq', 1),
            leverage_sample_ratio=self.parameters.get('leverage_sample_ratio', 0.5)
        )]

    def _get_lora_vi_models(self):
        """Create LoRa-VI model."""
        bias = self.parameters.get('bias', 0.0)
        return [LoRaVI(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            epsilon=self.parameters['epsilon'],
            alpha=self.parameters['alpha'],
            gamma=self.parameters['gamma'],
            decay=self.parameters.get('decay', 1.0),
            k=self.parameters['k'],
            bias=bias,
            max_inner_iterations=self.parameters.get('max_inner_iterations', 1),
            leverage_sample_ratio=self.parameters.get('leverage_sample_ratio', 0.5)
        )]

    def _get_dqn_models(self):
        """Create DQN model with paper architecture."""
        return [DQN(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            arch=self.parameters.get('arch', [100]),
            lr=self.parameters.get('lr', 1e-3),
            gamma=self.parameters.get('gamma', 0.99),
            epsilon=self.parameters.get('epsilon', 1.0),
            epsilon_decay=self.parameters.get('epsilon_decay', 0.999999),
            min_epsilon=self.parameters.get('min_epsilon', 0.01),
            batch_size=self.parameters.get('batch_size', 32),
            buffer_size=self.parameters.get('buffer_size', 100000),
            warmup_steps=self.parameters.get('warmup_steps', 500),
            update_freq=self.parameters.get('update_freq', 1),
            target_update_freq=self.parameters.get('target_update_freq', 100)
        )]

    def _get_sac_discrete_models(self):
        """Create Discrete SAC model with paper architecture."""
        return [DiscreteSAC(
            env=self.env,
            discretizer=self.discretizer,
            episodes=self.parameters['episodes'],
            max_steps=self.parameters['max_steps'],
            arch=self.parameters.get('arch', [100]),
            actor_lr=self.parameters.get('actor_lr', 3e-4),
            critic_lr=self.parameters.get('critic_lr', 3e-4),
            alpha_lr=self.parameters.get('alpha_lr', 3e-4),
            gamma=self.parameters.get('gamma', 0.99),
            tau=self.parameters.get('tau', 0.005),
            init_temperature=self.parameters.get('init_temperature', 0.2),
            learnable_temperature=self.parameters.get('learnable_temperature', True),
            batch_size=self.parameters.get('batch_size', 64),
            buffer_size=self.parameters.get('buffer_size', 100000),
            warmup_steps=self.parameters.get('warmup_steps', 500),
            update_freq=self.parameters.get('update_freq', 1)
        )]

    def _get_models_from_checkpoints(self):
        """Load models from checkpoints."""
        models = []
        checkpoint_dir = 'nn_checkpoints'
        if os.path.exists(checkpoint_dir):
            for path in os.listdir(checkpoint_dir):
                with open(os.path.join(checkpoint_dir, path), 'rb') as f:
                    model = pickle.load(f)
                models.append(model)
                os.remove(os.path.join(checkpoint_dir, path))
        return models

    def run_experiment(self, learner):
        """Run experiment for a single learner."""
        try:
            if learner is None:
                print("Warning: learner object is None")
                return None

            print(f"Starting training for model type: {type(learner).__name__}")
            print(f"Training parameters: episodes={learner.episodes}, max_steps={learner.max_steps}")

            learner.train(run_greedy_frequency=self.run_freq)

            required_attrs = ['training_steps', 'training_cumulative_reward', 
                            'greedy_steps', 'greedy_cumulative_reward']
            for attr in required_attrs:
                if not hasattr(learner, attr) or getattr(learner, attr) is None:
                    print(f"Warning: model is missing or has empty attribute {attr}")
                    return None

            print(f"Training complete: training_steps={len(learner.training_steps)}, "
                  f"training_cumulative_reward={len(learner.training_cumulative_reward)}, "
                  f"greedy_steps={len(learner.greedy_steps)}, "
                  f"greedy_cumulative_reward={len(learner.greedy_cumulative_reward)}")
            return learner

        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            import traceback
            print(f"Detailed traceback:\n{traceback.format_exc()}")
            return None

    def run_experiments(self, window=10):
        """Run experiments for all models and save results."""
        try:
            models = [self.run_experiment(model) for model in self.models]
            
            models = [model for model in models if model is not None]
            if not models:
                print("Warning: All model trainings failed")
                return

            all_training_steps = []
            all_training_rewards = []
            all_greedy_steps = []
            all_greedy_rewards = []
            
            for model in models:
                if hasattr(model, 'training_steps') and hasattr(model, 'training_cumulative_reward') and \
                   hasattr(model, 'greedy_steps') and hasattr(model, 'greedy_cumulative_reward'):
                    all_training_steps.append(model.training_steps)
                    all_training_rewards.append(model.training_cumulative_reward)
                    all_greedy_steps.append(model.greedy_steps)
                    all_greedy_rewards.append(model.greedy_cumulative_reward)
                else:
                    print(f"Warning: Model is missing required attributes")

            if all_training_steps and all_training_rewards and all_greedy_steps and all_greedy_rewards:
                training_steps = np.median(all_training_steps, axis=0).tolist()
                training_rewards = np.median(all_training_rewards, axis=0).tolist()
                greedy_steps = np.median(all_greedy_steps, axis=0).tolist()
                greedy_rewards = np.median(all_greedy_rewards, axis=0).tolist()
                
                data = {
                    'training_steps': training_steps,
                    'training_cumulative_reward': training_rewards,
                    'greedy_steps': greedy_steps,
                    'greedy_cumulative_reward': greedy_rewards
                }
            else:
                data = {
                    'training_steps': [],
                    'training_cumulative_reward': [],
                    'greedy_steps': [],
                    'greedy_cumulative_reward': []
                }

            # Create results directory if needed
            os.makedirs('results', exist_ok=True)
            
            with open(f'results/{self.name}', 'w') as f:
                json.dump(data, f)

            # Clean up checkpoints
            if os.path.exists('nn_checkpoints'):
                for path in os.listdir('nn_checkpoints'):
                    os.remove(os.path.join('nn_checkpoints', path))
                    
        except Exception as e:
            print(f"An error occurred while running experiments: {str(e)}")
            raise


def calculate_parameter_count(model_type, discretizer, params):
    """
    Calculate parameter count for a given model type.
    
    Returns: (total_params, details_dict)
    """
    num_states = int(np.prod(discretizer.n_states))
    num_actions = int(np.prod(discretizer.n_actions))
    dimensions = list(discretizer.n_states) + list(discretizer.n_actions)
    
    model_type = model_type.lower().replace('_', '-')
    
    if model_type in ['teql', 'tlr', 'original-tlr', 'convergent-tlr']:
        # Tensor decomposition: sum(d_i * k) for each dimension
        k = params.get('k', 10)
        total_params = sum(d * k for d in dimensions)
        details = {
            'method': 'CP Tensor Decomposition',
            'rank_k': k,
            'dimensions': dimensions,
            'formula': f'Î£(d_i Ã— k) = {sum(dimensions)} Ã— {k}'
        }
    
    elif model_type in ['lora-vi']:
        # Matrix decomposition: (|S| + |A|) * k
        k = params.get('k', 10)
        total_params = (num_states + num_actions) * k
        details = {
            'method': 'Low-Rank Matrix Decomposition',
            'rank_k': k,
            'num_states': num_states,
            'num_actions': num_actions,
            'formula': f'(|S| + |A|) Ã— k = ({num_states} + {num_actions}) Ã— {k}'
        }
    
    elif model_type == 'dqn':
        # Paper Architecture DQN: Direct bucket indices input to MLP
        if 'arch' in params:
            # NEW: Paper architecture with bucket indices as input
            arch = params.get('arch', [100])
            n_state_dims = len(discretizer.n_states)  # Number of state dimensions
            
            # Calculate MLP parameters
            total_params = 0
            prev_dim = n_state_dims
            
            for h_dim in arch:
                total_params += prev_dim * h_dim + h_dim  # weights + bias
                prev_dim = h_dim
            
            # Output layer
            total_params += prev_dim * num_actions + num_actions
            
            details = {
                'method': 'DQN (Paper Architecture - Bucket Indices)',
                'input_dims': n_state_dims,
                'hidden_layers': arch,
                'num_actions': num_actions,
                'total_params': total_params,
                'formula': f'MLP({n_state_dims} inputs  {arch}  {num_actions} actions）'
            }
        else:
            # OLD: Embedding-based architecture (backward compatibility)
            embed_dim = params.get('embed_dim', 32)
            hidden_dim = params.get('hidden_dim', 64)
            
            embed_params = num_states * embed_dim
            fc1_params = embed_dim * hidden_dim + hidden_dim
            fc2_params = hidden_dim * hidden_dim + hidden_dim
            fc3_params = hidden_dim * num_actions + num_actions
            fc_total = fc1_params + fc2_params + fc3_params
            total_params = embed_params + fc_total
            
            details = {
                'method': 'Tabular DQN (State Embedding)',
                'num_states': num_states,
                'num_actions': num_actions,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'embedding_params': embed_params,
                'fc_params': fc_total,
                'formula': f'|S| Ã— embed + FC = {num_states} Ã— {embed_dim} + {fc_total}'
            }
    
    elif model_type in ['sac-discrete', 'discrete-sac', 'sac']:
        # Paper Architecture SAC: Direct bucket indices input to MLP (Actor + Twin Critics)
        if 'arch' in params:
            # NEW: Paper architecture with bucket indices as input
            arch = params.get('arch', [100])
            n_state_dims = len(discretizer.n_states)
            learnable_temperature = params.get('learnable_temperature', True)
            
            # Calculate single network params
            network_params = 0
            prev_dim = n_state_dims
            
            for h_dim in arch:
                network_params += prev_dim * h_dim + h_dim
                prev_dim = h_dim
            
            network_params += prev_dim * num_actions + num_actions
            
            # Actor + 2 Critics + Temperature
            actor_total = network_params
            critic_total = network_params
            alpha_params = 1 if learnable_temperature else 0
            total_params = actor_total + 2 * critic_total + alpha_params
            
            details = {
                'method': 'SAC (Paper Architecture - Bucket Indices)',
                'input_dims': n_state_dims,
                'hidden_layers': arch,
                'num_actions': num_actions,
                'actor_params': actor_total,
                'critic_params_each': critic_total,
                'critic_params': 2 * critic_total,  # Total for both critics
                'alpha_params': alpha_params,
                'twin_critics': True,
                'learnable_temperature': learnable_temperature,
                'formula': f'Actor({actor_total}) + 2×Critic({critic_total}) + Alpha({alpha_params})'
            }
        else:
            # OLD: Embedding-based architecture (backward compatibility)
            embed_dim = params.get('embed_dim', 32)
            hidden_dim = params.get('hidden_dim', 64)
            learnable_temperature = params.get('learnable_temperature', True)
            
            # Actor
            actor_embed = num_states * embed_dim
            actor_fc1 = embed_dim * hidden_dim + hidden_dim
            actor_fc2 = hidden_dim * hidden_dim + hidden_dim
            actor_fc3 = hidden_dim * num_actions + num_actions
            actor_total = actor_embed + actor_fc1 + actor_fc2 + actor_fc3
            
            # Critic (x2 for twin Q-networks)
            critic_embed = num_states * embed_dim
            critic_fc1 = embed_dim * hidden_dim + hidden_dim
            critic_fc2 = hidden_dim * hidden_dim + hidden_dim
            critic_fc3 = hidden_dim * num_actions + num_actions
            critic_total = critic_embed + critic_fc1 + critic_fc2 + critic_fc3
            
            # Temperature parameter
            alpha_params = 1 if learnable_temperature else 0
            
            total_params = actor_total + 2 * critic_total + alpha_params
            
            details = {
                'method': 'Tabular Discrete SAC (State Embedding)',
                'num_states': num_states,
                'num_actions': num_actions,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'actor_params': actor_total,
                'critic_params_each': critic_total,
                'critic_params': 2 * critic_total,  # Total for both critics
                'alpha_params': alpha_params,
                'twin_critics': True,
                'learnable_temperature': learnable_temperature,
                'formula': f'Actor({actor_total}) + 2×Critic({critic_total}) + Alpha({alpha_params})'
            }
    
    else:
        total_params = 0
        details = {'method': 'Unknown'}
    
    return total_params, details


def get_algorithm_class(model_type):
    """Get algorithm class by model type string."""
    model_type = model_type.lower().replace('_', '-')
    
    mapping = {
        'original-tlr': OriginalTLR,
        'tlr': OriginalTLR,
        'convergent-tlr': TEQL,
        'teql': TEQL,
        'lora-vi': LoRaVI,
        'dqn': DQN,
        'sac-discrete': DiscreteSAC,
        'discrete-sac': DiscreteSAC,
        'sac': DiscreteSAC,
    }
    
    if model_type not in mapping:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(mapping.keys())}")
    
    return mapping[model_type]


if __name__ == "__main__":
    # Example usage and parameter comparison
    print("Parameter Count Comparison Example")
    print("="*50)
    
    class MockDiscretizer:
        def __init__(self):
            self.n_states = np.array([10, 10, 20, 20])
            self.n_actions = np.array([10])
            self.dimensions = np.array([10, 10, 20, 20, 10])
            self.discrete_action = False
    
    discretizer = MockDiscretizer()
    
    test_params = {
        'k': 10,
        'embed_dim': 32,
        'hidden_dim': 64
    }
    
    for model_type in ['teql', 'lora-vi', 'dqn', 'sac-discrete']:
        total, details = calculate_parameter_count(model_type, discretizer, test_params)
        print(f"\n{model_type.upper()}:")
        print(f"  Method: {details['method']}")
        print(f"  Formula: {details.get('formula', 'N/A')}")
        print(f"  Total: {total:,} parameters")