import os
import json
import pickle
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd

from src.algorithms.mlr_learning import MatrixLowRankLearning
from src.algorithms.Countmlr_learning import CountMatrixLowRankLearning
from src.utils.utils import Discretizer

class MLRExperiment:
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

    def _get_discretizer(self):
        states_structure = self.parameters.get('states_structure', None)
        discrete_action = self.parameters.get('discrete_action', False)
        return Discretizer(
            min_points_states=self.parameters['min_points_states'],
            max_points_states=self.parameters['max_points_states'],
            bucket_states=self.parameters['bucket_states'],
            min_points_actions=self.parameters['min_points_actions'],
            max_points_actions=self.parameters['max_points_actions'],
            bucket_actions=self.parameters['bucket_actions'],
            states_structure=states_structure,
            discrete_action=discrete_action
        )
    
    def _get_models(self):
        if self.parameters['type'] == 'mlr-model':
            return self._get_mlr_models()
        elif self.parameters['type'] == 'count-mlr-model':
            return self._get_count_mlr_models()
        return None

    def _get_models_from_checkpoints(self):
        models = []
        for path in os.listdir('nn_checkpoints'):
            with open(os.path.join('nn_checkpoints', path), 'rb') as f:
                model = pickle.load(f)

            if len(model.training_steps) > len(model.greedy_steps):
                model.training_steps.pop()
                model.training_cumulative_reward.pop()
            elif len(model.training_steps) > model.episode:
                model.episode += 1

            models.append(model)
            os.remove(os.path.join('nn_checkpoints', path))
        return models

    def _get_mlr_models(self):
        # 为k参数列表中的每个值创建一个模型
        k_values = self.parameters['k']
        if not isinstance(k_values, list):
            k_values = [k_values]
            
        models = []
        for k in k_values:
            for _ in range(self.nodes // len(k_values)):  # 平均分配nodes给每个k值
                models.append(MatrixLowRankLearning(
                    env=self.env,
                    discretizer=self.discretizer,
                    episodes=self.parameters['episodes'],
                    max_steps=self.parameters['max_steps'],
                    epsilon=self.parameters['epsilon'],
                    alpha=self.parameters['alpha'],
                    gamma=self.parameters['gamma'],
                    decay=self.parameters['decay'],
                    k=k  # 传入单个k值而不是整个列表
                ))
        return models

    def _get_count_mlr_models(self):
        # 同样处理k参数列表
        k_values = self.parameters['k']
        if not isinstance(k_values, list):
            k_values = [k_values]
            
        models = []
        for k in k_values:
            for _ in range(self.nodes // len(k_values)):
                models.append(CountMatrixLowRankLearning(
                    env=self.env,
                    discretizer=self.discretizer,
                    episodes=self.parameters['episodes'],
                    max_steps=self.parameters['max_steps'],
                    epsilon=self.parameters['epsilon'],
                    alpha=self.parameters['alpha'],
                    gamma=self.parameters['gamma'],
                    decay=self.parameters['decay'],
                    k=k,
                    c=self.parameters.get('c', 8.5)
                ))
        return models

    def run_experiment(self, learner):
        learner.train(run_greedy_frequency=self.run_freq)
        return learner

    def run_experiments(self, window):
        with Pool(self.nodes) as pool:
            models = pool.map(self.run_experiment, self.models)

        steps = []
        max_length = self.parameters['episodes']
        for model in models:
            s = model.greedy_steps
            if s:  # 确保有步数数据
                steps.append(pd.Series(s).rolling(window).median())

        # 检查steps列表是否为空
        if steps:
            steps = np.nanmedian(steps, axis=0)
            # 确保steps是可迭代的
            if isinstance(steps, (np.float64, float)):
                steps = [steps]
        else:
            steps = []  # 如果没有有效数据，返回空列表

        # 过滤掉None和nan值
        rewards = [learner.mean_reward for learner in models if learner.mean_reward is not None]
        rewards = [r for r in rewards if not np.isnan(r)]

        data = {'steps': list(steps), 'rewards': rewards}

        # 生成结果文件名
        result_filename = f'results/{self.name}'

        # 保存结果
        with open(result_filename, 'w') as f:
            json.dump(data, f)

        # 清理检查点文件
        for path in os.listdir('nn_checkpoints'):
            os.remove(os.path.join('nn_checkpoints', path))
    
    # def run_experiments(self, window):
    #     with Pool(self.nodes) as pool:
    #         models = pool.map(self.run_experiment, self.models)

    #     steps = []
    #     max_length = self.parameters['episodes']
    #     for model in models:
    #         s = model.greedy_steps
    #         steps.append(pd.Series(s).rolling(window).median())

    #     steps = np.median(steps, axis=0)
    #     rewards = [learner.mean_reward for learner in models]
    #     data = {'steps': list(steps), 'rewards': rewards}

    #     with open(f'results/{self.name}', 'w') as f:
    #         json.dump(data, f)

    #     for path in os.listdir('nn_checkpoints'):
    #         os.remove(os.path.join('nn_checkpoints', path))
            
    def measure_mean_runtime(self):
        state = self.env.reset()
        if isinstance(state, tuple):  # Handle new gym API
            state = state[0]
        action = self.models[0].choose_action(state)
        state_prime, reward, done, _ = self.env.step(action)

        start_time = time.time()
        for _ in range(100_000):
            self.models[0].update_q_matrix(state, action, state_prime, reward, done)
        end_time = time.time()
        return end_time - start_time