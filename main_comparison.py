"""
Main Experiment: Compare All Algorithms
========================================
Compare tensor-based methods (TEQL, TLR, LoRa-VI) with deep learning 
baselines (DQN, SAC) in high-dimensional discrete state-action spaces.

Usage:
    # Run TEQL on cartpole
    python main_comparison.py --env cartpole --model teql
    
    # Compare with DQN and SAC
    python main_comparison.py --env cartpole --model teql dqn sac
    
    # Run highway environment
    python main_comparison.py --env highway --model teql dqn sac
    
    # Run all environments
    python main_comparison.py --env all --model dqn sac
    
    # Show parameter counts
    python main_comparison.py --env cartpole --params-only

Available models: teql, tlr, lora-vi, dqn, sac
Available envs: cartpole, pendulum, highway, all
"""

import os
import sys
import json
import shutil
import numpy as np
import argparse

# Add source directories to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'experiments'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'environments'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'utils'))

from src.experiments.experiment_runner_all import Experiment, calculate_parameter_count
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.environments.pendulum import CustomPendulumEnv
from src.environments.highway_wrapper import CustomHighwayEnv
from src.utils.utils import Discretizer


N_ITERATIONS = 30

# Configuration mapping
CONFIGS = {
    'cartpole': {
        'teql': 'cartpole_convergent_tlr_learning.json',
        'tlr': 'cartpole_original_tlr_learning.json',
        'lora-vi': 'cartpole_lora_vi_learning.json',
        'dqn': 'cartpole_dqn.json',
        'sac': 'cartpole_sac.json'
    },
    'pendulum': {
        'teql': 'pendulum_convergent_tlr_learning.json',
        'tlr': 'pendulum_original_tlr_learning.json',
        'lora-vi': 'pendulum_lora_vi_learning.json',
        'dqn': 'pendulum_dqn.json',
        'sac': 'pendulum_sac.json'
    },
    'highway': {
        'teql': 'highway_convergent_tlr_learning.json',
        'tlr': 'highway_original_tlr_learning.json',
        'dqn': 'highway_dqn.json',
        'sac': 'highway_sac.json'
    }
}

# Environment instances (lazy loading for highway to avoid import errors if not installed)
env_cartpole = CustomContinuousCartPoleEnv()
env_pendulum = CustomPendulumEnv()

def get_highway_env():
    """Lazy load highway environment to avoid import errors if highway_env not installed."""
    try:
        return CustomHighwayEnv()
    except Exception as e:
        print(f"Warning: Could not create highway environment: {e}")
        print("Please install highway_env: pip install highway-env")
        return None

ENVS = {
    'cartpole': env_cartpole,
    'pendulum': env_pendulum,
    'highway': None  # Will be lazy loaded
}

# Window sizes for each environment (matching main.py)
WINDOWS = {
    'cartpole': 50,
    'pendulum': 30,
    'highway': 30
}


def get_environment(env_name):
    """Create environment by name."""
    if env_name == 'highway':
        # Lazy load highway environment
        if ENVS.get('highway') is None:
            ENVS['highway'] = get_highway_env()
    return ENVS.get(env_name)


def create_discretizer(params):
    """Create discretizer from parameters."""
    states_structure = params.get('states_structure', None)
    discrete_action = params.get('discrete_action', False)
    
    bucket_states = np.array(params['bucket_states'], dtype=np.int32)
    bucket_actions = np.array(params['bucket_actions'], dtype=np.int32)
    
    discretizer = Discretizer(
        min_points_states=params['min_points_states'],
        max_points_states=params['max_points_states'],
        bucket_states=bucket_states,
        min_points_actions=params['min_points_actions'],
        max_points_actions=params['max_points_actions'],
        bucket_actions=bucket_actions,
        states_structure=states_structure,
        discrete_action=discrete_action
    )
    
    discretizer.n_states = np.array(discretizer.n_states, dtype=np.int32)
    discretizer.n_actions = np.array(discretizer.n_actions, dtype=np.int32)
    discretizer.dimensions = np.concatenate([discretizer.n_states, discretizer.n_actions]).astype(np.int32)
    
    return discretizer


def load_config(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def calculate_parameter_summary(env_name, config_dir='parameters'):
    """Calculate and display parameter counts for all algorithms."""
    env = get_environment(env_name)
    if env is None:
        return
    
    configs = CONFIGS.get(env_name, {})
    
    print(f"\n{'='*70}")
    print(f"Parameter Count Summary for {env_name.upper()}")
    print(f"{'='*70}")
    
    results = []
    
    for model_type, config_file in configs.items():
        config_path = os.path.join(config_dir, config_file)
        if os.path.exists(config_path):
            params = load_config(config_path)
            discretizer = create_discretizer(params)
            
            total_params, details = calculate_parameter_count(model_type, discretizer, params)
            results.append((model_type, total_params, details))
        else:
            print(f"  [SKIP] {model_type}: config not found at {config_path}")
    
    if not results:
        print("No valid configurations found!")
        return
    
    # Sort by parameter count
    results.sort(key=lambda x: x[1])
    
    # Print results
    for model_type, total_params, details in results:
        print(f"\n{model_type.upper()}:")
        print(f"  Method:      {details['method']}")
        print(f"  Formula:     {details.get('formula', 'N/A')}")
        print(f"  Total:       {total_params:,} parameters")
    
    # Print comparison table
    print(f"\n{'-'*70}")
    print(f"{'Algorithm':<20} {'Parameters':>15} {'Ratio to TEQL':>20}")
    print(f"{'-'*70}")
    
    teql_params = None
    for model_type, total_params, _ in results:
        if model_type == 'teql':
            teql_params = total_params
            break
    
    for model_type, total_params, _ in results:
        if teql_params and teql_params > 0:
            ratio = total_params / teql_params
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"
        print(f"{model_type.upper():<20} {total_params:>15,} {ratio_str:>20}")
    
    print(f"{'='*70}\n")


def backup_and_clear_results(iteration, env_name, models):
    """Backup results to iteration-specific directory - matching main.py style."""
    backup_dir = f'results_backup/iteration_{iteration}'
    os.makedirs(backup_dir, exist_ok=True)
    
    results_dir = 'results'
    if os.path.exists(results_dir):
        for model_type in models:
            config_file = CONFIGS.get(env_name, {}).get(model_type)
            if config_file:
                src_path = os.path.join(results_dir, config_file)
                dst_path = os.path.join(backup_dir, config_file)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)


def get_result_filename(env_name, model_type):
    """Get the result filename for a given model type."""
    return CONFIGS.get(env_name, {}).get(model_type)


def get_existing_iterations():
    """Get list of all existing iteration numbers."""
    backup_base = 'results_backup'
    if not os.path.exists(backup_base):
        return []
    
    iterations = []
    for d in os.listdir(backup_base):
        if d.startswith('iteration_'):
            try:
                iterations.append(int(d.split('_')[1]))
            except ValueError:
                pass
    return sorted(iterations)


def find_missing_iterations(env_name, model_type, target_iterations):
    """
    Find which iterations are missing results for a given model.
    
    Args:
        env_name: Environment name
        model_type: Model type (e.g., 'dqn', 'sac', 'teql')
        target_iterations: Total number of iterations we want
        
    Returns:
        List of iteration numbers that need to be run for this model
    """
    backup_base = 'results_backup'
    config_file = get_result_filename(env_name, model_type)
    
    if not config_file:
        print(f"Warning: No config file found for {model_type}")
        return list(range(target_iterations))
    
    missing = []
    for i in range(target_iterations):
        filepath = os.path.join(backup_base, f'iteration_{i}', config_file)
        if not os.path.exists(filepath):
            missing.append(i)
    
    return missing


def find_all_missing_iterations(env_name, models, target_iterations):
    """
    Find missing iterations for all models.
    
    Returns:
        dict: {model_type: [list of missing iteration numbers]}
    """
    missing_map = {}
    for model_type in models:
        missing = find_missing_iterations(env_name, model_type, target_iterations)
        missing_map[model_type] = missing
    return missing_map


def print_missing_summary(env_name, models, target_iterations):
    """Print a summary of which iterations are missing for each model."""
    missing_map = find_all_missing_iterations(env_name, models, target_iterations)
    
    print(f"\n{'='*70}")
    print(f"Missing Iterations Summary for {env_name.upper()}")
    print(f"{'='*70}")
    print(f"Target iterations: {target_iterations}")
    print(f"{'-'*70}")
    
    for model_type in models:
        missing = missing_map[model_type]
        existing = target_iterations - len(missing)
        
        if len(missing) == 0:
            status = "✓ Complete"
            missing_str = "None"
        elif len(missing) == target_iterations:
            status = "✗ No results"
            missing_str = "All"
        else:
            status = f"△ Partial ({existing}/{target_iterations})"
            # Summarize missing iterations
            if len(missing) <= 10:
                missing_str = str(missing)
            else:
                missing_str = f"{missing[:5]}...{missing[-3:]} ({len(missing)} total)"
        
        print(f"{model_type.upper():<12} {status:<25} Missing: {missing_str}")
    
    print(f"{'='*70}\n")
    return missing_map


def collect_results_from_backup(env_name, models):
    """Collect all iteration results for aggregation - matching main.py style."""
    all_results = {model: [] for model in models}
    
    backup_base = 'results_backup'
    
    if not os.path.exists(backup_base):
        return all_results
    
    backup_dirs = sorted(
        [d for d in os.listdir(backup_base) if d.startswith('iteration_')],
        key=lambda x: int(x.split('_')[1])
    )
    
    for backup_dir in backup_dirs:
        for model_type in models:
            config_file = CONFIGS.get(env_name, {}).get(model_type)
            if config_file:
                filepath = os.path.join(backup_base, backup_dir, config_file)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        all_results[model_type].append(data)
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
    
    return all_results


def run_single_model_iteration(iteration, env_name, model_type, config_dir='parameters', run_freq=10):
    """Run a single model for a specific iteration."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('nn_checkpoints', exist_ok=True)
    
    env = get_environment(env_name)
    window = WINDOWS.get(env_name, 30)
    
    config_file = CONFIGS.get(env_name, {}).get(model_type)
    if config_file:
        print(f"\nRunning {model_type.upper()}...")
        try:
            experiment = Experiment(config_file, env, run_freq=run_freq, param_dir=config_dir)
            experiment.run_experiments(window=window)
            
            # Backup this model's result
            backup_dir = f'results_backup/iteration_{iteration}'
            os.makedirs(backup_dir, exist_ok=True)
            
            src_path = os.path.join('results', config_file)
            dst_path = os.path.join(backup_dir, config_file)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                print(f"  ✓ Saved to {dst_path}")
            
        except Exception as e:
            print(f"Error running {model_type}: {e}")
            import traceback
            traceback.print_exc()


def run_iteration(iteration, env_name, models, config_dir='parameters', run_freq=10):
    """Run one iteration of experiments for all specified models - matching main.py style."""
    print(f"\n{'='*60}")
    print(f"Running iteration {iteration + 1}")
    print(f"{'='*60}")
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('nn_checkpoints', exist_ok=True)
    
    env = get_environment(env_name)
    window = WINDOWS.get(env_name, 30)
    
    for model_type in models:
        config_file = CONFIGS.get(env_name, {}).get(model_type)
        if config_file:
            print(f"\nRunning {model_type.upper()}...")
            try:
                experiment = Experiment(config_file, env, run_freq=run_freq, param_dir=config_dir)
                experiment.run_experiments(window=window)
            except Exception as e:
                print(f"Error running {model_type}: {e}")
                import traceback
                traceback.print_exc()
    
    # Backup results
    backup_and_clear_results(iteration, env_name, models)


def run_comparison_experiment(env_name, models, n_iterations=1, config_dir='parameters', run_freq=10):
    """
    Run comparison experiment for specified algorithms with multiple iterations.
    Smart detection: finds missing iterations for each model and fills them in.
    """
    os.makedirs('results', exist_ok=True)
    os.makedirs('results_backup', exist_ok=True)
    
    # Print missing iterations summary
    missing_map = print_missing_summary(env_name, models, n_iterations)
    
    # Check if any work needs to be done
    total_missing = sum(len(missing) for missing in missing_map.values())
    if total_missing == 0:
        print("All iterations complete! No new experiments needed.")
    else:
        print(f"Total experiments to run: {total_missing}")
        
        # Run missing iterations for each model
        for model_type in models:
            missing_iterations = missing_map[model_type]
            
            if not missing_iterations:
                print(f"\n{model_type.upper()}: Already complete, skipping.")
                continue
            
            print(f"\n{'#'*60}")
            print(f"# Running {model_type.upper()} - {len(missing_iterations)} iterations needed")
            print(f"# Iterations: {missing_iterations[:10]}{'...' if len(missing_iterations) > 10 else ''}")
            print(f"{'#'*60}")
            
            for idx, iteration in enumerate(missing_iterations):
                print(f"\n[{model_type.upper()}] Iteration {iteration + 1}/{n_iterations} "
                      f"(Progress: {idx + 1}/{len(missing_iterations)})")
                run_single_model_iteration(iteration, env_name, model_type, config_dir, run_freq)
    
    # Collect and aggregate results
    all_results = collect_results_from_backup(env_name, models)
    
    # Save aggregated results
    aggregated = {}
    for model_type, results_list in all_results.items():
        if results_list:
            training_rewards = np.array([r['training_cumulative_reward'] for r in results_list])
            greedy_rewards = np.array([r['greedy_cumulative_reward'] for r in results_list])
            
            aggregated[model_type] = {
                'n_iterations': len(results_list),
                'training_reward_mean': np.mean(training_rewards, axis=0).tolist(),
                'training_reward_std': np.std(training_rewards, axis=0).tolist(),
                'greedy_reward_mean': np.mean(greedy_rewards, axis=0).tolist(),
                'greedy_reward_std': np.std(greedy_rewards, axis=0).tolist(),
            }
    
    # Save aggregated results
    with open(f'results/{env_name}_aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for model_type, data in aggregated.items():
        if 'greedy_reward_mean' in data and len(data['greedy_reward_mean']) > 0:
            final_mean = np.mean(data['greedy_reward_mean'][-10:])
            final_std = np.mean(data['greedy_reward_std'][-10:])
            print(f"{model_type.upper():15} | Final Reward: {final_mean:>8.2f} ± {final_std:.2f} "
                  f"({data['n_iterations']} iterations)")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Run algorithm comparison experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TEQL on cartpole
  python main_comparison.py --env cartpole --model teql
  
  # Compare TEQL, DQN, and SAC
  python main_comparison.py --env cartpole --model teql dqn sac
  
  # Run on all environments
  python main_comparison.py --env all --model teql dqn sac
  
  # Show parameter counts
  python main_comparison.py --env cartpole --params-only
  
  # Check missing iterations without running
  python main_comparison.py --env cartpole --model dqn sac --check-only
  
  # Multiple iterations (smart fill-in)
  python main_comparison.py --env cartpole --iterations 30

Smart Iteration Detection:
  The script automatically detects which iterations are missing for each model.
  If you previously ran 20 iterations of SAC and now want to add DQN,
  running '--model dqn --iterations 20' will fill in DQN for iterations 0-19.

Available models: teql, tlr, lora-vi, dqn, sac
Available envs: cartpole, pendulum, highway, all
        """
    )
    parser.add_argument('--env', type=str, default='cartpole', 
                        choices=['cartpole', 'pendulum', 'highway', 'all'],
                        help='Environment to test')
    parser.add_argument('--config-dir', type=str, default='parameters',
                        help='Directory containing config files')
    parser.add_argument('--run-freq', type=int, default=10,
                        help='Frequency of greedy evaluation')
    parser.add_argument('--iterations', type=int, default=N_ITERATIONS,
                        help='Number of iterations to run')
    parser.add_argument('--params-only', action='store_true',
                        help='Only show parameter counts')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check missing iterations, do not run experiments')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                        choices=['teql', 'tlr', 'lora-vi', 'dqn', 'sac'],
                        help='Model(s) to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all available models')
    
    args = parser.parse_args()
    
    # Determine which environments to run
    if args.env == 'all':
        envs = ['cartpole', 'pendulum', 'highway']
    else:
        envs = [args.env]
    
    # Determine which models to run
    if args.model:
        models = args.model
    elif args.all:
        models = ['teql', 'tlr', 'lora-vi', 'dqn', 'sac']
    else:
        models = ['teql', 'dqn', 'sac']
    
    if args.params_only:
        for env_name in envs:
            calculate_parameter_summary(env_name, args.config_dir)
    elif args.check_only:
        # Just check and display missing iterations
        for env_name in envs:
            print(f"\n{'#'*70}")
            print(f"# ENVIRONMENT: {env_name.upper()}")
            print(f"{'#'*70}")
            print_missing_summary(env_name, models, args.iterations)
    else:
        all_env_results = {}
        for env_name in envs:
            print(f"\n{'#'*70}")
            print(f"# ENVIRONMENT: {env_name.upper()}")
            print(f"{'#'*70}")
            
            results = run_comparison_experiment(
                env_name,
                models=models,
                n_iterations=args.iterations,
                config_dir=args.config_dir,
                run_freq=args.run_freq
            )
            all_env_results[env_name] = results
        
        # Save final results (matching main.py style)
        if all_env_results:
            with open('final_results.json', 'w') as f:
                json.dump(all_env_results, f, indent=2)
        
        print("\nExperiments completed. Results saved.")


if __name__ == "__main__":
    main()