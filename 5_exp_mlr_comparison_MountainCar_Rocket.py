import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.environments.goddard import CustomGoddardEnv
from src.experiments.experiment_MLR import MLRExperiment
from src.utils.utils import OOMFormatter

# Initialize environments
env_mountaincar = CustomContinuous_MountainCarEnv()
env_rocket = CustomGoddardEnv()

N_NODES = 1

def run_single_experiment(env, name, window):
    """运行单个实验并包含错误处理"""
    try:
        print(f"\nInitializing experiment: {name}")
        experiment = MLRExperiment(name, env, N_NODES)
        print("Starting experiment run...")
        experiment.run_experiments(window=window)
        print(f"Successfully completed experiment: {name}")
        return True
    except Exception as e:
        print(f"Error in experiment {name}: {str(e)}")
        return False

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Mountain Car实验
    experiments = [f for f in os.listdir('parameters') if 'mountaincar' in f and 'mlr' in f]
    experiments = sorted(experiments)  # 确保顺序一致
    
    print("\nPreparing to run Mountain Car experiments:")
    for exp in experiments:
        print(f"- {exp}")
        
    for name in experiments:
        success = run_single_experiment(env_mountaincar, name, window=100)
        if not success:
            print(f"Skipping remaining experiments due to error in {name}")
            break
            
    # 检查实验结果
    results_files = [f for f in os.listdir('results') if 'mountaincar' in f]
    print("\nAvailable results files:")
    for f in results_files:
        print(f"- {f}")

    # 如果mountain car实验成功，继续运行rocket实验
    if results_files:
        print("\nStarting Rocket experiments...")
        experiments = [f for f in os.listdir('parameters') if 'rocket' in f and 'mlr' in f]
        for name in experiments:
            run_single_experiment(env_rocket, name, window=50)

    # 检查是否有足够的结果来绘图
    mountain_results = [f for f in os.listdir('results') if 'mountaincar' in f and 'mlr' in f]
    rocket_results = [f for f in os.listdir('results') if 'rocket' in f and 'mlr' in f]

    if not (mountain_results and rocket_results):
        print("Not enough results to generate plots")
        exit()

    print("\nGenerating plots...")

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

    # Mountain car steps
    prefix = 'results/mountaincar_'
    steps = range(0, 5000, 10)
    axes[0].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g', label=labels[0])
    axes[0].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r', label=labels[1])
    axes[0].set_xlabel("Episodes", labelpad=4)
    axes[0].set_ylabel("(a) Number of Steps")
    axes[0].set_yticks([0, 5000, 10000])
    axes[0].set_xticks([0, 2500, 5000])
    axes[0].yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].set_xlim(0, 5000)
    axes[0].legend(fontsize=12, loc='lower right')

    # Rocket steps
    prefix = 'results/rocket_'
    steps = range(0, 500000, 10)
    axes[1].plot(steps, json.load(open(prefix + 'mlr_learning.json', 'r'))['steps'], color='g', label=labels[0])
    axes[1].plot(steps, json.load(open(prefix + 'count_mlr.json', 'r'))['steps'], color='r', label=labels[1])
    axes[1].set_xlabel("Episodes", labelpad=4)
    axes[1].set_ylabel("(b) Number of Steps")
    axes[1].set_yticks([0, 500, 1000])
    axes[1].set_xticks([0, 250000, 500000])
    axes[1].yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1].set_xlim(0, 500000)
    axes[1].legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    fig.savefig('figures/mlr_comparison_rest_steps.jpg', dpi=300, bbox_inches='tight')

    # Plot rewards
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes = axes.flatten()

    # Mountain car rewards
    prefix = 'results/mountaincar_'
    rewards = [
        json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
        json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
    ]
    axes[0].boxplot(rewards)
    axes[0].set_ylabel("(a) Cumulative Reward")
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_yticks([0, 50, 100])
    axes[0].set_ylim(0, 100)
    axes[0].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Rocket rewards
    prefix = 'results/rocket_'
    rewards = [
        json.load(open(prefix + 'mlr_learning.json', 'r'))['rewards'],
        json.load(open(prefix + 'count_mlr.json', 'r'))['rewards'],
    ]
    axes[1].boxplot(rewards)
    axes[1].set_ylabel("(b) Cumulative Reward")
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_yticks([0, 70, 140])
    axes[1].yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    fig.savefig('figures/mlr_comparison_rest_rewards.jpg', dpi=300, bbox_inches='tight')

    print("\nExperiments completed. Results saved in figures/")