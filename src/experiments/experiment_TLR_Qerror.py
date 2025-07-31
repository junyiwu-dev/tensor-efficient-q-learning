import os
import json
import pickle
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd

from src.algorithms.tlr_learning_original_only import TensorLowRankLearning as OriginalTLR
from src.algorithms.tlr_learning_convergent_with_qerror_UCB_penalty import TensorLowRankLearning as ConvergentTLR
from src.utils.utils import Discretizer

class TLRExperiment:
    def __init__(self, name, env, nodes, recover=False, run_freq=10):
        self.env = env
        self.nodes = nodes
        self.name = name
        self.run_freq = run_freq

        with open(f'parameters/{name}', 'r') as f:
            self.parameters = json.load(f)

        self.discretizer = self._get_discretizer()

        if recover:
            self.models = self._get_models_from_checkpoints()
        else:
            self.models = self._get_models()

        # 移除了 q_error_dir 的创建和相关代码

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
        ) for _ in range(self.nodes)]

    def _get_convergent_tlr_models(self):
        bias = self.parameters.get('bias', 0.0)
        return [ConvergentTLR(
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
        ) for _ in range(self.nodes)]

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
                print("警告: learner 对象为空")
                return None
            
            print(f"开始训练模型，类型: {type(learner).__name__}")
            print(f"训练参数: episodes={learner.episodes}, max_steps={learner.max_steps}")
            
            # 注意：Q_error仍然在算法内部计算和使用（EUGE需要），但不保存历史到文件
            learner.train(run_greedy_frequency=self.run_freq)
            
            required_attrs = ['training_steps', 'training_cumulative_reward', 'greedy_steps', 'greedy_cumulative_reward']
            for attr in required_attrs:
                if not hasattr(learner, attr) or getattr(learner, attr) is None:
                    print(f"警告: 模型缺少或属性为空 {attr}")
                    return None
            
            print(f"训练完成: training_steps={len(learner.training_steps)}, training_cumulative_reward={len(learner.training_cumulative_reward)}, "
                  f"greedy_steps={len(learner.greedy_steps)}, greedy_cumulative_reward={len(learner.greedy_cumulative_reward)}")
            return learner
            
        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return None

    # 移除了 _save_q_error 方法 - 不再保存Q_error历史到文件
    # 但注意：Q_error仍在算法内部维护，用于EUGE探索策略

    def run_experiments(self, window):
        try:
            with Pool(self.nodes) as pool:
                models = pool.map(self.run_experiment, self.models)
            
            models = [model for model in models if model is not None]
            if not models:
                print("警告: 所有模型训练都失败了")
                return

            # 收集所有模型的数据
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
                    print(f"警告: 模型缺少必要的属性")

            if all_training_steps and all_training_rewards and all_greedy_steps and all_greedy_rewards:
                # 计算中位数
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
            print(f"运行实验时发生错误: {str(e)}")
            raise