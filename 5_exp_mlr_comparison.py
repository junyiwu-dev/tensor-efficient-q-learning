import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv

from src.experiments.experiment_MLR import MLRExperiment
from src.utils.utils import OOMFormatter

# Initialize environments
env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
env_mountaincar = CustomContinuous_MountainCarEnv()
env_rocket = CustomGoddardEnv()

N_NODES = 1

        
if __name__ == "__main__":
    # Debug prints
    print("\nAvailable parameter files:")
    print([f for f in os.listdir('parameters') if 'mlr' in f])
    
    print("\nPendulum experiments to run:")
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f and 'mlr' in f]
    print(experiments)
    
    print("\nTrying to read first experiment parameters:")
    if experiments:
        with open(f'parameters/{experiments[0]}', 'r') as f:
            params = json.load(f)
        print("Parameters for first experiment:", params)

    # 尝试运行一个实验
    print("\nStarting first experiment...")
    if experiments:
        experiment = MLRExperiment(experiments[0], env_pendulum, N_NODES)
        print("Experiment initialized, starting training...")
        experiment.run_experiments(window=30)
        print("First experiment completed")
    # Pendulum
    experiments = [f for f in os.listdir('parameters') if 'pendulum' in f and 'mlr' in f]
    experiments_done = [f for f in os.listdir('results') if 'pendulum' in f]
    for name in tqdm(experiments, desc="Pendulum Experiments"):
        if name in experiments_done:
            continue
        experiment = MLRExperiment(name, env_pendulum, N_NODES)
        experiment.run_experiments(window=30)

    # Cartpole
    experiments = [f for f in os.listdir('parameters') if 'cartpole' in f and 'mlr' in f]
    experiments_done = [f for f in os.listdir('results') if 'cartpole' in f]
    for name in tqdm(experiments, desc="Cartpole Experiments"):
        if name in experiments_done:
            continue
        experiment = MLRExperiment(name, env_cartpole, N_NODES)
        experiment.run_experiments(window=50)

    # Mountaincar
    experiments = [f for f in os.listdir('parameters') if 'mountaincar' in f and 'mlr' in f]
    experiments_done = [f for f in os.listdir('results') if 'mountaincar' in f]
    for name in tqdm(experiments, desc="Mountain Car Experiments"):
        if name in experiments_done:
            continue
        experiment = MLRExperiment(name, env_mountaincar, N_NODES)
        experiment.run_experiments(window=50)

    # Rocket
    experiments = [f for f in os.listdir('parameters') if 'rocket' in f and 'mlr' in f]
    experiments_done = [f for f in os.listdir('results') if 'rocket' in f]
    for name in tqdm(experiments, desc="Rocket Experiments"):
        if name in experiments_done:
            continue
        experiment = MLRExperiment(name, env_rocket, N_NODES)
        experiment.run_experiments(window=50)

    # Plotting results
    labels = ["MLR-lear.", "Count MLR-lear."]

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 18})

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
        axes = axes.flatten()

        # Pendulum
        prefix = 'results/pendulum_'
        steps = range(0, 15000, 10)
        axes[0].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[0].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r')
        axes[0].set_xlabel("Episodes", labelpad=4)
        axes[0].set_ylabel("(a) $\#$ Steps")
        axes[0].set_xlim(0, 15000)
        axes[0].set_yticks([0, 50, 100])
        axes[0].set_xticks([0, 7500, 15000])
        axes[0].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
        axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[0].legend(labels, fontsize=12, loc='lower right')
        axes[0].grid()

        # Cartpole
        prefix = 'results/cartpole_'
        steps = range(0, 40000, 10)
        axes[1].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[1].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r')
        axes[1].set_xlabel("Episodes", labelpad=4)
        axes[1].set_ylabel("(b) $\#$ Steps")
        axes[1].set_yticks([0, 50, 100])
        axes[1].set_xticks([0, 20000, 40000])
        axes[1].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
        axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[1].set_xlim(0, 40000)
        axes[1].legend(labels, fontsize=12, loc='lower right')
        axes[1].grid()

        # Mountain Car
        prefix = 'results/mountaincar_'
        steps = range(0, 5000, 10)
        axes[2].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[2].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r')
        axes[2].set_xlabel("Episodes", labelpad=4)
        axes[2].set_ylabel("(c) $\#$ Steps")
        axes[2].set_yticks([0, 5000, 10000])
        axes[2].set_xticks([0, 2500, 5000])
        axes[2].yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
        axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[2].set_xlim(0, 5000)
        axes[2].legend(labels, fontsize=12)
        axes[2].grid()

        # Rocket
        prefix = 'results/rocket_'
        steps = range(0, 500000, 10)
        axes[3].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g')
        axes[3].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r')
        axes[3].set_xlabel("Episodes", labelpad=4)
        axes[3].set_ylabel("(d) $\#$ Steps")
        axes[3].set_yticks([0, 500, 1000])
        axes[3].set_xticks([0, 250000, 500000])
        axes[3].yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[3].set_xlim(0, 500000)
        axes[3].legend(labels, fontsize=12)
        axes[3].grid()

        plt.tight_layout()
        fig.savefig('figures/mlr_comparison_steps.jpg', dpi=300)

    # Plot rewards
    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 16})

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[8, 7])
        axes = axes.flatten()

        # Pendulum rewards
        prefix = 'results/pendulum_'
        rewards = [
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
        ]
        axes[0].boxplot(rewards)
        axes[0].set_ylabel("(a) Cumm. Reward")
        axes[0].set_xticklabels(labels, rotation=90, size=12)
        axes[0].set_yticks([0, 50, 100])
        axes[0].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
        axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Cartpole rewards
        prefix = 'results/cartpole_'
        rewards = [
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
        ]
        axes[1].boxplot(rewards)
        axes[1].set_ylabel("(b) Cumm. Reward")
        axes[1].set_xticklabels(labels, rotation=90, size=12)
        axes[1].set_yticks([-100, 0, 100])
        axes[1].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
        axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Mountain Car rewards
        prefix = 'results/mountaincar_'
        rewards = [
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
        ]
        axes[2].boxplot(rewards)
        axes[2].set_ylabel("(c) Cumm. Reward")
        axes[2].set_xticklabels(labels, rotation=90, size=12)
        axes[2].set_yticks([0, 50, 100])
        axes[2].set_ylim(0, 100)
        axes[2].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
        axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Rocket rewards
        prefix = 'results/rocket_'
        rewards = [
            json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
            json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
        ]
        axes[3].boxplot(rewards)
        axes[3].set_ylabel("(d) Cumm. Reward")
        axes[3].set_xticklabels(labels, rotation=90, size=12)
        axes[3].set_yticks([0, 70, 140])
        axes[3].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
        axes[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.tight_layout()
        fig.savefig('figures/mlr_comparison_rewards.jpg', dpi=300)

    print("\nExperiments completed. Results saved in figures/")