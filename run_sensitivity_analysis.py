import os
import json
import shutil
from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.experiments.experiment_runner import Experiment

N_ITERATIONS = 100

DISCRETIZATION_LEVELS = {
    'very_coarse': {
        'pendulum': {'bucket_states': [8, 8], 'bucket_actions': [4]},
        'cartpole': {'bucket_states': [5, 5, 8, 8], 'bucket_actions': [4]}
    },
    'coarse': {
        'pendulum': {'bucket_states': [15, 15], 'bucket_actions': [8]},
        'cartpole': {'bucket_states': [8, 8, 15, 15], 'bucket_actions': [8]}
    },
    'current': {
        'pendulum': {'bucket_states': [20, 20], 'bucket_actions': [10]},
        'cartpole': {'bucket_states': [10, 10, 20, 20], 'bucket_actions': [10]}
    },
    'fine': {
        'pendulum': {'bucket_states': [30, 30], 'bucket_actions': [15]},
        'cartpole': {'bucket_states': [15, 15, 25, 25], 'bucket_actions': [15]}
    },
    'very_fine': {
        'pendulum': {'bucket_states': [40, 40], 'bucket_actions': [20]},
        'cartpole': {'bucket_states': [20, 20, 40, 40], 'bucket_actions': [20]}
    }
}

def run_sensitivity_experiments():
    
    environments = ['pendulum', 'cartpole']
    
    for level_name, level_config in DISCRETIZATION_LEVELS.items():
        print(f"\n{'='*60}")
        print(f"Processing discretization level: {level_name}")
        print(f"{'='*60}")
        
        # Create results directory - use the original project's naming style
        results_dir = f'results_backup_{level_name}'
        os.makedirs(results_dir, exist_ok=True)
        
        for env_name in environments:
            # Select environment
            env = CustomPendulumEnv() if env_name == 'pendulum' else CustomContinuousCartPoleEnv()

            # Read base configuration
            base_config_file = f'parameters/{env_name}_convergent_tlr_learning.json'
            with open(base_config_file, 'r') as f:
                base_config = json.load(f)
            

            # Update discretization parameters
            base_config['bucket_states'] = level_config[env_name]['bucket_states']
            base_config['bucket_actions'] = level_config[env_name]['bucket_actions']
            
            # Run TLR (original)
            print(f"\nRunning {env_name} - TLR - {level_name}")
            tlr_config = base_config.copy()
            tlr_config['type'] = 'original-tlr'
            
            run_experiments_for_config(
                tlr_config, env, env_name, 'original', 
                results_dir, N_ITERATIONS
            )
            
            # Run TEQL (convergent with penalty)
            print(f"\nRunning {env_name} - TEQL - {level_name}")
            teql_config = base_config.copy()
            teql_config['type'] = 'convergent-tlr'
            teql_config['lambda_penalty'] = 0.01
            teql_config['epsilon_penalty'] = 1e-6
            
            run_experiments_for_config(
                teql_config, env, env_name, 'convergent', 
                results_dir, N_ITERATIONS
            )
    
    print("\nSensitivity analysis experiments completed!")

def run_experiments_for_config(config, env, env_name, algo_type, results_dir, n_iterations):
    
    # Create temporary config file
    temp_config_filename = f'temp_{env_name}_{algo_type}.json'  # Just filename
    temp_config_path = f'parameters/{temp_config_filename}'     # Full path for file operations
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run experiments n_iterations times
    for iteration in range(n_iterations):
        if iteration % 10 == 0:
            print(f"  Iteration {iteration + 1}/{n_iterations}")

        # Run experiment - pass only filename
        experiment = Experiment(temp_config_filename, env, run_freq=10)  # Only filename!
        experiment.run_experiments(window=30 if env_name == 'pendulum' else 50)

        # Backup results
        backup_dir = f'{results_dir}/iteration_{iteration}'
        os.makedirs(backup_dir, exist_ok=True)

        result_file = f'results/{temp_config_filename}'  # Results will be in results/ directory
        if os.path.exists(result_file):
            target_file = os.path.join(backup_dir, f'{env_name}_{algo_type}_tlr_learning.json')
            shutil.move(result_file, target_file)
    
    # Clean up temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    
def visualize_results():
    print("\nGenerating sensitivity analysis visualizations...")

    # Import and call the visualization function
    from visualization.discretization_sensitivity import main as plot_sensitivity

    # Run visualization
    plot_sensitivity()

    print("Visualization complete!")

def main():
    print("TEQL Discretization Sensitivity Analysis")
    print("-"*80)
    
    run_sensitivity_experiments()

    visualize_results()
    
    print("Results saved in:")
    for level in DISCRETIZATION_LEVELS.keys():
        print(f"  - results_backup_{level}/")
    print("Figures saved in: figures/")

if __name__ == "__main__":
    main()