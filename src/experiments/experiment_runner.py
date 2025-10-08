import os
import json
import pickle
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd

from src.algorithms.tlr_original import TensorLowRankLearning as OriginalTLR  
from src.algorithms.teql import TensorEfficientQL as TEQL
from src.utils.utils import Discretizer

class Experiment:
    def __init__(self, name, env, recover=False, run_freq=10):
        self.env = env
        self.name = name
        self.run_freq = run_freq

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

        self.discretizer = self._get_discretizer()

        if recover:
            self.models = self._get_models_from_checkpoints()
        else:
            self.models = self._get_models()

    def _get_discretizer(self):
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
        
        if model_type == 'original-tlr':
            return self._get_original_tlr_models()
        elif model_type == 'convergent-tlr':
            return self._get_convergent_tlr_models()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _get_original_tlr_models(self):
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

    def _get_convergent_tlr_models(self):
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
            max_inner_iterations=self.parameters.get('max_inner_iterations', 100)
        )] 

    def _get_models_from_checkpoints(self):
        models = []
        for path in os.listdir('nn_checkpoints'):
            with open(os.path.join('nn_checkpoints', path), 'rb') as f:
                model = pickle.load(f)
            models.append(model)
            os.remove(os.path.join('nn_checkpoints', path))
        return models

    def run_experiment(self, learner):
        try:
            if learner is None:
                print("Warning: learner object is None")
                return None

            print(f"Starting training for model type: {type(learner).__name__}")
            print(f"Training parameters: episodes={learner.episodes}, max_steps={learner.max_steps}")

            learner.train(run_greedy_frequency=self.run_freq)

            required_attrs = ['training_steps', 'training_cumulative_reward', 'greedy_steps', 'greedy_cumulative_reward']
            for attr in required_attrs:
                if not hasattr(learner, attr) or getattr(learner, attr) is None:
                    print(f"Warning: model is missing or has empty attribute {attr}")
                    return None

            print(f"Training complete: training_steps={len(learner.training_steps)}, training_cumulative_reward={len(learner.training_cumulative_reward)}, "
                  f"greedy_steps={len(learner.greedy_steps)}, greedy_cumulative_reward={len(learner.greedy_cumulative_reward)}")
            return learner

        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            import traceback
            print(f"Detailed traceback:\n{traceback.format_exc()}")
            return None


    def run_experiments(self, window):
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

            with open(f'results/{self.name}', 'w') as f:
                json.dump(data, f)

            for path in os.listdir('nn_checkpoints'):
                os.remove(os.path.join('nn_checkpoints', path))
        except Exception as e:
            print(f"An error occurred while running experiments: {str(e)}")
            raise