import os
import json
import shutil
from tqdm import tqdm
import numpy as np

from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.experiments.experiment_runner import Experiment
from src.utils.utils import OOMFormatter

env_pendulum = CustomPendulumEnv()
env_cartpole = CustomContinuousCartPoleEnv()
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



def run_iteration(iteration):
    backup_and_clear_results(iteration)
    os.makedirs('results', exist_ok=True)
    
    print("\nRunning Pendulum experiments:")
    print(f"Running Original TLR...")
    experiment = Experiment('pendulum_original_tlr_learning.json', env_pendulum)
    experiment.run_experiments(window=30)
    
    print(f"Running TEQL...")
    experiment = Experiment('pendulum_convergent_tlr_learning.json', env_pendulum)
    experiment.run_experiments(window=30)

    print("\nRunning Cartpole experiments:")
    print(f"Running Original TLR...")
    experiment = Experiment('cartpole_original_tlr_learning.json', env_cartpole)
    experiment.run_experiments(window=50)
    
    print(f"Running TEQL...")
    experiment = Experiment('cartpole_convergent_tlr_learning.json', env_cartpole)
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

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results_backup', exist_ok=True)
    
    start_iteration = 0
    while os.path.exists(f'results_backup/iteration_{start_iteration}'):
        start_iteration += 1
    
    print(f"Starting from iteration {start_iteration}")
    
    for iteration in range(start_iteration, N_ITERATIONS):
        print(f"\nRunning iteration {iteration + 1}/{N_ITERATIONS}")
        run_iteration(iteration)
    
    pendulum_results = collect_results_from_backup('pendulum')
    cartpole_results = collect_results_from_backup('cartpole')
    
    # Save collected results to final_results.json
    if pendulum_results or cartpole_results:
        with open('final_results.json', 'w') as f:
            json.dump({
                'pendulum': pendulum_results,
                'cartpole': cartpole_results
            }, f)
    
    print("\nExperiments completed. Results saved.") 
if __name__ == "__main__":
    main()