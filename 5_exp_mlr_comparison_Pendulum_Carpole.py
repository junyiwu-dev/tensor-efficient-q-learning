import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.experiments.experiment_MLR import MLRExperiment
from src.utils.utils import OOMFormatter

# Initialize environments
env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()

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

    # Plotting results
    labels = ["MLR-lear.", "Count MLR-lear."]

    # Set the plotting style
    plt.style.use('default')
    matplotlib.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'axes.labelsize': 16,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'lines.linewidth': 2,
        'figure.dpi': 300
    })

    # Plot steps
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes = axes.flatten()

    # Pendulum
    prefix = 'results/pendulum_'
    steps = range(0, 15000, 10)
    axes[0].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g', label=labels[0])
    axes[0].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r', label=labels[1])
    axes[0].set_xlabel("Episodes", labelpad=4)
    axes[0].set_ylabel("(a) Number of Steps")
    axes[0].set_xlim(0, 15000)
    axes[0].set_yticks([0, 50, 100])
    axes[0].set_xticks([0, 7500, 15000])
    axes[0].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].legend(fontsize=12, loc='lower right')

    # Cartpole
    prefix = 'results/cartpole_'
    steps = range(0, 40000, 10)
    axes[1].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g', label=labels[0])
    axes[1].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r', label=labels[1])
    axes[1].set_xlabel("Episodes", labelpad=4)
    axes[1].set_ylabel("(b) Number of Steps")
    axes[1].set_yticks([0, 50, 100])
    axes[1].set_xticks([0, 20000, 40000])
    axes[1].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1].set_xlim(0, 40000)
    axes[1].legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    fig.savefig('figures/mlr_comparison_steps.jpg', dpi=300, bbox_inches='tight')

    # Plot rewards
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes = axes.flatten()

    # Pendulum rewards
    prefix = 'results/pendulum_'
    rewards = [
        json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
        json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
    ]
    axes[0].boxplot(rewards)
    axes[0].set_ylabel("(a) Cumulative Reward")
    axes[0].set_xticklabels(labels, rotation=45)
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
    axes[1].set_ylabel("(b) Cumulative Reward")
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_yticks([-100, 0, 100])
    axes[1].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    fig.savefig('figures/mlr_comparison_rewards.jpg', dpi=300, bbox_inches='tight')

    print("\nExperiments completed. Results saved in figures/")