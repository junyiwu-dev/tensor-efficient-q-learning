"""
Unified Discretization Sensitivity Analysis (TEQL & TLR)
=========================================================
Analyze how different discretization levels affect TEQL and TLR performance.

This script runs TEQL and/or TLR with various bucket configurations to study
the trade-off between discretization granularity and learning performance.

Discretization Levels:
- very_fine: Finest granularity, most parameters
- fine: Standard granularity
- coarse: Reduced granularity
- very_coarse: Minimal granularity, fewest parameters

Usage:
    # Run TEQL sensitivity analysis (default)
    python sensitivity_analysis.py --algo teql
    
    # Run TLR sensitivity analysis
    python sensitivity_analysis.py --algo tlr
    
    # Run both algorithms
    python sensitivity_analysis.py --algo all

    # Run specific environment
    python sensitivity_analysis.py --env cartpole --algo teql
    python sensitivity_analysis.py --env pendulum --algo tlr
    
    # Run specific discretization level
    python sensitivity_analysis.py --level fine --algo teql
    python sensitivity_analysis.py --level very_fine coarse --algo tlr
    
    # Check missing iterations
    python sensitivity_analysis.py --check-only --algo all
    
    # Show parameter counts for each level
    python sensitivity_analysis.py --params-only --algo teql

    # Custom iterations
    python sensitivity_analysis.py --iterations 50 --algo tlr
"""

import os
import sys
import json
import shutil
import copy
import numpy as np
import argparse
from datetime import datetime

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
from src.utils.utils import Discretizer


# ============================================================================
# Configuration
# ============================================================================

N_ITERATIONS = 30
BACKUP_DIR = 'results_backup_sensitivity'

# Base config files for each algorithm
BASE_CONFIGS = {
    'teql': {
        'cartpole': 'parameters/cartpole_convergent_tlr_learning.json',
        'pendulum': 'parameters/pendulum_convergent_tlr_learning.json'
    },
    'tlr': {
        'cartpole': 'parameters/cartpole_original_tlr_learning.json',
        'pendulum': 'parameters/pendulum_original_tlr_learning.json'
    }
}

# Discretization levels for each environment
DISCRETIZATION_LEVELS = {
    'pendulum': {
        'very_fine': {
            'bucket_states': [80, 80],
            'bucket_actions': [40]
        },
        'fine': {
            'bucket_states': [30, 30],
            'bucket_actions': [15]
        },
        'coarse': {
            'bucket_states': [15, 15],
            'bucket_actions': [8]
        },
        'very_coarse': {
            'bucket_states': [8, 8],
            'bucket_actions': [4]
        }
    },
    'cartpole': {
        'very_fine': {
            'bucket_states': [40, 40, 80, 80],
            'bucket_actions': [40]
        },
        'fine': {
            'bucket_states': [15, 15, 30, 30],
            'bucket_actions': [15]
        },
        'coarse': {
            'bucket_states': [8, 8, 15, 15],
            'bucket_actions': [8]
        },
        'very_coarse': {
            'bucket_states': [5, 5, 8, 8],
            'bucket_actions': [4]
        }
    }
}

# Level order for display
LEVEL_ORDER = ['very_coarse', 'coarse', 'fine', 'very_fine']

# Environment instances
ENVS = {
    'cartpole': CustomContinuousCartPoleEnv(),
    'pendulum': CustomPendulumEnv()
}

# Window sizes
WINDOWS = {
    'cartpole': 50,
    'pendulum': 30
}

# Algorithm type mapping for parameter count calculation
ALGO_TYPE_MAP = {
    'teql': 'teql',
    'tlr': 'original-tlr'
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_base_config(env_name, algo):
    """Load base configuration file for a specific algorithm."""
    config_path = BASE_CONFIGS[algo][env_name]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Base config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def create_level_config(env_name, level, algo):
    """Create configuration for a specific discretization level and algorithm."""
    # Load base config
    config = load_base_config(env_name, algo)
    
    # Overwrite discretization parameters
    level_params = DISCRETIZATION_LEVELS[env_name][level]
    config['bucket_states'] = level_params['bucket_states']
    config['bucket_actions'] = level_params['bucket_actions']
    
    return config


def create_discretizer(params):
    """Create discretizer from parameters."""
    bucket_states = np.array(params['bucket_states'], dtype=np.int32)
    bucket_actions = np.array(params['bucket_actions'], dtype=np.int32)
    discrete_action = params.get('discrete_action', False)
    
    discretizer = Discretizer(
        min_points_states=params['min_points_states'],
        max_points_states=params['max_points_states'],
        bucket_states=bucket_states,
        min_points_actions=params['min_points_actions'],
        max_points_actions=params['max_points_actions'],
        bucket_actions=bucket_actions,
        discrete_action=discrete_action
    )
    
    discretizer.n_states = np.array(discretizer.n_states, dtype=np.int32)
    discretizer.n_actions = np.array(discretizer.n_actions, dtype=np.int32)
    discretizer.dimensions = np.concatenate([discretizer.n_states, discretizer.n_actions]).astype(np.int32)
    
    return discretizer


def get_result_filename(env_name, level, algo):
    """Get result filename for a specific env, level, and algorithm."""
    return f"{env_name}_{algo}_{level}.json"


def get_iteration_dir(iteration):
    """Get directory path for a specific iteration."""
    return os.path.join(BACKUP_DIR, f'iteration_{iteration}')


def check_existing_results(env_name, level, algo, n_iterations):
    """Check which iterations are missing for a specific env/level/algo combination."""
    missing = []
    for i in range(n_iterations):
        result_path = os.path.join(get_iteration_dir(i), get_result_filename(env_name, level, algo))
        if not os.path.exists(result_path):
            missing.append(i)
    return missing


def calculate_level_params(env_name, level, algo):
    """Calculate parameter count for a specific discretization level."""
    config = create_level_config(env_name, level, algo)
    discretizer = create_discretizer(config)
    algo_type = ALGO_TYPE_MAP[algo]
    total_params, details = calculate_parameter_count(algo_type, discretizer, config)
    return total_params, details, config


# ============================================================================
# Display Functions
# ============================================================================

def print_params_summary(algos):
    """Print parameter count summary for all levels."""
    for algo in algos:
        print("\n" + "=" * 80)
        print(f"{algo.upper()} Parameter Count by Discretization Level")
        print("=" * 80)
        
        for env_name in ['cartpole', 'pendulum']:
            print(f"\n{env_name.upper()}")
            print("-" * 80)
            print(f"{'Level':<15} {'Bucket States':<25} {'Bucket Actions':<15} {'Parameters':>12}")
            print("-" * 80)
            
            for level in LEVEL_ORDER:
                total_params, details, config = calculate_level_params(env_name, level, algo)
                bucket_states = config['bucket_states']
                bucket_actions = config['bucket_actions']
                
                print(f"{level:<15} {str(bucket_states):<25} {str(bucket_actions):<15} {total_params:>12,}")
            
            print("-" * 80)
        
        print("\n")


def print_missing_summary(envs, levels, algos, n_iterations):
    """Print summary of missing iterations."""
    for algo in algos:
        print("\n" + "=" * 80)
        print(f"Missing Iterations Summary ({algo.upper()})")
        print(f"Target iterations: {n_iterations}")
        print("=" * 80)
        
        for env_name in envs:
            print(f"\n{env_name.upper()}")
            print("-" * 80)
            
            for level in levels:
                missing = check_existing_results(env_name, level, algo, n_iterations)
                completed = n_iterations - len(missing)
                
                if len(missing) == 0:
                    status = "✓ Complete"
                    missing_str = "None"
                elif len(missing) == n_iterations:
                    status = "✗ No results"
                    missing_str = "All"
                else:
                    status = f"△ Partial ({completed}/{n_iterations})"
                    if len(missing) <= 10:
                        missing_str = str(missing)
                    else:
                        missing_str = f"[{missing[0]}...{missing[-1]}] ({len(missing)} total)"
                
                print(f"  {level:<15} {status:<25} Missing: {missing_str}")
        
        print("=" * 80 + "\n")


# ============================================================================
# Experiment Running
# ============================================================================

def run_single_experiment(env_name, level, iteration, algo):
    """Run a single experiment for the specified algorithm."""
    print(f"\n{'='*60}")
    print(f"Running: {env_name.upper()} | Level: {level} | Iteration: {iteration} | Algorithm: {algo.upper()}")
    print(f"{'='*60}")
    
    # Create configuration
    config = create_level_config(env_name, level, algo)
    
    # Calculate and display parameters
    discretizer = create_discretizer(config)
    algo_type = ALGO_TYPE_MAP[algo]
    total_params, _ = calculate_parameter_count(algo_type, discretizer, config)
    print(f"Bucket states: {config['bucket_states']}")
    print(f"Bucket actions: {config['bucket_actions']}")
    print(f"Parameters: {total_params:,}")
    print(f"Algorithm type: {config['type']}")
    
    # Get environment
    env = ENVS[env_name]
    window = WINDOWS[env_name]
    
    # Create temporary config file in parameters/ directory (where Experiment expects it)
    temp_config_filename = f'temp_{env_name}_{level}_{algo}_config.json'
    temp_config_path = os.path.join('parameters', temp_config_filename)
    
    # Ensure parameters directory exists
    os.makedirs('parameters', exist_ok=True)
    
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run experiment - Experiment expects just the filename, not full path
        experiment = Experiment(temp_config_filename, env)
        experiment.run_experiments(window=window)
        
        # Results are saved to results/{temp_config_filename} by Experiment
        temp_result_path = os.path.join('results', temp_config_filename)
        
        if not os.path.exists(temp_result_path):
            print(f"✗ Error: Results file not found at {temp_result_path}")
            return False
        
        # Read the results
        with open(temp_result_path, 'r') as f:
            results = json.load(f)
        
        # Save results to our backup directory
        iteration_dir = get_iteration_dir(iteration)
        os.makedirs(iteration_dir, exist_ok=True)
        
        result_filename = get_result_filename(env_name, level, algo)
        result_path = os.path.join(iteration_dir, result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved results to {result_path}")
        
        # Clean up the temp result file
        if os.path.exists(temp_result_path):
            os.remove(temp_result_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def run_sensitivity_analysis(envs, levels, algos, n_iterations):
    """Run full sensitivity analysis for specified algorithms."""
    for algo in algos:
        print("\n" + "#" * 80)
        print(f"# {algo.upper()} Discretization Sensitivity Analysis")
        print("#" * 80)
        print(f"Environments: {', '.join(envs)}")
        print(f"Levels: {', '.join(levels)}")
        print(f"Iterations: {n_iterations}")
        print(f"Backup directory: {BACKUP_DIR}")
        print("#" * 80 + "\n")
        
        # Create backup directory
        os.makedirs(BACKUP_DIR, exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Track progress
        total_experiments = 0
        completed_experiments = 0
        skipped_experiments = 0
        failed_experiments = 0
        
        for env_name in envs:
            for level in levels:
                # Check missing iterations
                missing = check_existing_results(env_name, level, algo, n_iterations)
                
                if not missing:
                    print(f"[SKIP] {env_name}/{level}/{algo}: All {n_iterations} iterations complete")
                    skipped_experiments += n_iterations
                    continue
                
                print(f"\n[RUN] {env_name}/{level}/{algo}: Running {len(missing)} missing iterations")
                
                for iteration in missing:
                    total_experiments += 1
                    success = run_single_experiment(env_name, level, iteration, algo)
                    
                    if success:
                        completed_experiments += 1
                    else:
                        failed_experiments += 1
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"{algo.upper()} SENSITIVITY ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Completed:  {completed_experiments}")
        print(f"Skipped:    {skipped_experiments}")
        print(f"Failed:     {failed_experiments}")
        print(f"Total runs: {total_experiments}")
        print("=" * 80 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Discretization Sensitivity Analysis (TEQL & TLR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TEQL sensitivity analysis (default)
  python sensitivity_analysis.py --algo teql
  
  # Run TLR sensitivity analysis
  python sensitivity_analysis.py --algo tlr

  # Run both algorithms
  python sensitivity_analysis.py --algo all
  
  # Run specific environment
  python sensitivity_analysis.py --env cartpole --algo teql
  
  # Run specific levels
  python sensitivity_analysis.py --level fine coarse --algo tlr

  # Check what's missing
  python sensitivity_analysis.py --check-only --algo all
  
  # Show parameter counts
  python sensitivity_analysis.py --params-only --algo teql

Discretization Levels (from coarsest to finest):
  very_coarse -> coarse -> fine -> very_fine
        """
    )
    
    parser.add_argument('--env', type=str, nargs='+',
                        choices=['cartpole', 'pendulum', 'all'],
                        default=['all'],
                        help='Environment(s) to test')
    parser.add_argument('--level', type=str, nargs='+',
                        choices=['very_fine', 'fine', 'coarse', 'very_coarse', 'all'],
                        default=['all'],
                        help='Discretization level(s) to test')
    parser.add_argument('--algo', type=str, nargs='+',
                        choices=['teql', 'tlr', 'all'],
                        default=['teql'],
                        help='Algorithm(s) to run: teql, tlr, or all')
    parser.add_argument('--iterations', type=int, default=N_ITERATIONS,
                        help='Number of iterations per configuration')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check missing iterations')
    parser.add_argument('--params-only', action='store_true',
                        help='Only show parameter counts')
    
    args = parser.parse_args()
    
    # Determine environments
    if 'all' in args.env:
        envs = ['cartpole', 'pendulum']
    else:
        envs = args.env
    
    # Determine levels
    if 'all' in args.level:
        levels = LEVEL_ORDER
    else:
        levels = args.level
    
    # Determine algorithms
    if 'all' in args.algo:
        algos = ['teql', 'tlr']
    else:
        algos = args.algo
    
    # Execute requested action
    if args.params_only:
        print_params_summary(algos)
    elif args.check_only:
        print_missing_summary(envs, levels, algos, args.iterations)
    else:
        run_sensitivity_analysis(envs, levels, algos, args.iterations)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()