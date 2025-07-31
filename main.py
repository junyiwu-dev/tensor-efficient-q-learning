import os
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.experiments.experiment_TLR_Qerror import TLRExperiment
from src.utils.utils import OOMFormatter

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
N_NODES = 1
N_ITERATIONS = 100

EXPERIMENT_CONFIGS = {
    'pendulum': {
        'original': 'pendulum_original_tlr_learning.json',
        'convergent': 'pendulum_convergent_tlr_learning.json',
        'window': 30,
        'env': env_pendulum
    },
    'cartpole': {
        'original': 'cartpole_original_tlr_learning.json',
        'convergent': 'cartpole_convergent_tlr_learning.json',
        'window': 50,
        'env': env_cartpole
    }
}

def backup_and_clear_results(iteration):
    backup_dir = f'results_backup/iteration_{iteration}'
    os.makedirs(backup_dir, exist_ok=True)
    
    results_dir = 'results'
    if os.path.exists(results_dir):
        for env_config in EXPERIMENT_CONFIGS.values():
            for result_file in [env_config['original'], env_config['convergent']]:
                src_path = os.path.join(results_dir, result_file)
                dst_path = os.path.join(backup_dir, result_file)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)

def collect_results_from_backup(env_prefix):
    all_results = []
    
    backup_dirs = sorted([d for d in os.listdir('results_backup') if d.startswith('iteration_')],
                        key=lambda x: int(x.split('_')[1]))
    
    for backup_dir in backup_dirs:
        try:
            with open(os.path.join('results_backup', backup_dir, f'{env_prefix}_original_tlr_learning.json'), 'r') as f:
                original_data = json.load(f)
            with open(os.path.join('results_backup', backup_dir, f'{env_prefix}_convergent_tlr_learning.json'), 'r') as f:
                convergent_data = json.load(f)
            all_results.append([original_data, convergent_data])
        except Exception as e:
            print(f"Error reading from {backup_dir}: {e}")
            continue
            
    return all_results

def plot_metrics(ax_steps, ax_rewards, results, title, run_greedy_frequency=10):
    if not results:
        return
    
    try:
        training_steps_original = []
        training_steps_convergent = []
        training_rewards_original = []
        training_rewards_convergent = []
        greedy_steps_original = []
        greedy_steps_convergent = []
        greedy_rewards_original = []
        greedy_rewards_convergent = []
        
        for data in results:
            if all(k in data[0] for k in ['training_steps', 'training_cumulative_reward', 'greedy_steps', 'greedy_cumulative_reward']):
                training_steps_original.append(data[0]['training_steps'])
                training_rewards_original.append(data[0]['training_cumulative_reward'])
                greedy_steps_original.append(data[0]['greedy_steps'])
                greedy_rewards_original.append(data[0]['greedy_cumulative_reward'])
            if all(k in data[1] for k in ['training_steps', 'training_cumulative_reward', 'greedy_steps', 'greedy_cumulative_reward']):
                training_steps_convergent.append(data[1]['training_steps'])
                training_rewards_convergent.append(data[1]['training_cumulative_reward'])
                greedy_steps_convergent.append(data[1]['greedy_steps'])
                greedy_rewards_convergent.append(data[1]['greedy_cumulative_reward'])

        if training_steps_original and training_steps_convergent:
            # 转换为 numpy 数组
            training_steps_original = np.array(training_steps_original)
            training_steps_convergent = np.array(training_steps_convergent)
            training_rewards_original = np.array(training_rewards_original)
            training_rewards_convergent = np.array(training_rewards_convergent)
            greedy_steps_original = np.array(greedy_steps_original)
            greedy_steps_convergent = np.array(greedy_steps_convergent)
            greedy_rewards_original = np.array(greedy_rewards_original)
            greedy_rewards_convergent = np.array(greedy_rewards_convergent)
            
            # x 轴：训练模式为每个 episode，贪婪模式为每 run_greedy_frequency 个 episode
            episodes = np.arange(1, training_steps_original.shape[1] + 1)
            greedy_episodes = np.arange(run_greedy_frequency, training_steps_original.shape[1] + 1, run_greedy_frequency)
            
            # 绘制 steps
            ax_steps.plot(episodes, np.mean(training_steps_original, axis=0), label='Original TLR Train Steps', color='blue', linestyle='--')
            ax_steps.plot(episodes, np.mean(training_steps_convergent, axis=0), label='Convergent TLR Train Steps', color='red', linestyle='--')
            ax_steps.plot(greedy_episodes, np.mean(greedy_steps_original, axis=0), label='Original TLR Greedy Steps', color='blue')
            ax_steps.plot(greedy_episodes, np.mean(greedy_steps_convergent, axis=0), label='Convergent TLR Greedy Steps', color='red')
            ax_steps.fill_between(episodes,
                               np.mean(training_steps_original, axis=0) - np.std(training_steps_original, axis=0),
                               np.mean(training_steps_original, axis=0) + np.std(training_steps_original, axis=0),
                               alpha=0.2, color='blue')
            ax_steps.fill_between(episodes,
                               np.mean(training_steps_convergent, axis=0) - np.std(training_steps_convergent, axis=0),
                               np.mean(training_steps_convergent, axis=0) + np.std(training_steps_convergent, axis=0),
                               alpha=0.2, color='red')
            ax_steps.fill_between(greedy_episodes,
                               np.mean(greedy_steps_original, axis=0) - np.std(greedy_steps_original, axis=0),
                               np.mean(greedy_steps_original, axis=0) + np.std(greedy_steps_original, axis=0),
                               alpha=0.2, color='blue')
            ax_steps.fill_between(greedy_episodes,
                               np.mean(greedy_steps_convergent, axis=0) - np.std(greedy_steps_convergent, axis=0),
                               np.mean(greedy_steps_convergent, axis=0) + np.std(greedy_steps_convergent, axis=0),
                               alpha=0.2, color='red')
            ax_steps.set_title(f'{title} - Steps per Episode')
            ax_steps.set_xlabel("Episodes")
            ax_steps.set_ylabel("Number of Steps")
            ax_steps.legend()
            ax_steps.grid(True)

            # 绘制 rewards
            ax_rewards.plot(episodes, np.mean(training_rewards_original, axis=0), label='Original TLR Train Rewards', color='blue', linestyle='--')
            ax_rewards.plot(episodes, np.mean(training_rewards_convergent, axis=0), label='Convergent TLR Train Rewards', color='red', linestyle='--')
            ax_rewards.plot(greedy_episodes, np.mean(greedy_rewards_original, axis=0), label='Original TLR Greedy Rewards', color='blue')
            ax_rewards.plot(greedy_episodes, np.mean(greedy_rewards_convergent, axis=0), label='Convergent TLR Greedy Rewards', color='red')
            ax_rewards.fill_between(episodes,
                                 np.mean(training_rewards_original, axis=0) - np.std(training_rewards_original, axis=0),
                                 np.mean(training_rewards_original, axis=0) + np.std(training_rewards_original, axis=0),
                                 alpha=0.2, color='blue')
            ax_rewards.fill_between(episodes,
                                 np.mean(training_rewards_convergent, axis=0) - np.std(training_rewards_convergent, axis=0),
                                 np.mean(training_rewards_convergent, axis=0) + np.std(training_rewards_convergent, axis=0),
                                 alpha=0.2, color='red')
            ax_rewards.fill_between(greedy_episodes,
                                 np.mean(greedy_rewards_original, axis=0) - np.std(greedy_rewards_original, axis=0),
                                 np.mean(greedy_rewards_original, axis=0) + np.std(greedy_rewards_original, axis=0),
                                 alpha=0.2, color='blue')
            ax_rewards.fill_between(greedy_episodes,
                                 np.mean(greedy_rewards_convergent, axis=0) - np.std(greedy_rewards_convergent, axis=0),
                                 np.mean(greedy_rewards_convergent, axis=0) + np.std(greedy_rewards_convergent, axis=0),
                                 alpha=0.2, color='red')
            ax_rewards.set_title(f'{title} - Rewards per Episode')
            ax_rewards.set_xlabel("Episodes")
            ax_rewards.set_ylabel("Cumulative Reward")
            ax_rewards.legend()
            ax_rewards.grid(True)
            
    except Exception as e:
        print(f"Error plotting metrics for {title}: {str(e)}")

def run_iteration(iteration):
    backup_and_clear_results(iteration)
    os.makedirs('results', exist_ok=True)
    
    print("\nRunning Pendulum experiments:")
    print(f"Running Original TLR...")
    experiment = TLRExperiment('pendulum_original_tlr_learning.json', env_pendulum, N_NODES)
    experiment.run_experiments(window=30)
    
    print(f"Running Convergent TLR...")
    experiment = TLRExperiment('pendulum_convergent_tlr_learning.json', env_pendulum, N_NODES)
    experiment.run_experiments(window=30)

    print("\nRunning Cartpole experiments:")
    print(f"Running Original TLR...")
    experiment = TLRExperiment('cartpole_original_tlr_learning.json', env_cartpole, N_NODES)
    experiment.run_experiments(window=50)
    
    print(f"Running Convergent TLR...")
    experiment = TLRExperiment('cartpole_convergent_tlr_learning.json', env_cartpole, N_NODES)
    experiment.run_experiments(window=50)
    
    backup_dir = f'results_backup/iteration_{iteration}'
    os.makedirs(backup_dir, exist_ok=True)
    
    results_dir = 'results'
    if os.path.exists(results_dir):
        for env_name in ['pendulum', 'cartpole']:
            for result_file in [f'{env_name}_original_tlr_learning.json', f'{env_name}_convergent_tlr_learning.json']:
                src_path = os.path.join(results_dir, result_file)
                dst_path = os.path.join(backup_dir, result_file)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)

def setup_plotting_style():
    plt.style.use('default')
    matplotlib.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'lines.linewidth': 2,
        'figure.figsize': (15, 12)
    })

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results_backup', exist_ok=True)
    # 移除了 q_error_records 目录的创建 - 不再保存Q_error历史记录
    
    start_iteration = 0
    while os.path.exists(f'results_backup/iteration_{start_iteration}'):
        start_iteration += 1
    
    print(f"Starting from iteration {start_iteration}")
    
    for iteration in range(start_iteration, N_ITERATIONS):
        print(f"\nRunning iteration {iteration + 1}/{N_ITERATIONS}")
        run_iteration(iteration)
    
    pendulum_results = collect_results_from_backup('pendulum')
    cartpole_results = collect_results_from_backup('cartpole')
    
    if pendulum_results or cartpole_results:
        setup_plotting_style()
        fig, ((ax1_steps, ax1_rewards), (ax2_steps, ax2_rewards)) = plt.subplots(2, 2, figsize=(15, 12))
        
        plot_metrics(ax1_steps, ax1_rewards, pendulum_results, 'Pendulum', run_greedy_frequency=10)
        plot_metrics(ax2_steps, ax2_rewards, cartpole_results, 'Cartpole', run_greedy_frequency=10)
        
        plt.tight_layout()
        fig.savefig('aggregated_tlr_comparison_results.png', dpi=300, bbox_inches='tight')
        
        with open('final_results.json', 'w') as f:
            json.dump({
                'pendulum': pendulum_results,
                'cartpole': cartpole_results
            }, f)
    
    print("\nExperiments completed. Results saved.") # 移除了关于Q_error records的提示 - 不再保存历史记录

if __name__ == "__main__":
    main()