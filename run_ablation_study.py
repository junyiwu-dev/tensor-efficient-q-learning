import os
import json
import shutil
from src.environments.pendulum import CustomPendulumEnv
from src.environments.cartpole import CustomContinuousCartPoleEnv
from src.experiments.experiment_runner import Experiment

N_ITERATIONS = 100

def run_ablation_experiments():
    
    configs = [
        ('pendulum', 'convergent', 'results_backup_TEQL_wo_penalty'),  # Without penalty
        ('pendulum', 'convergent_penalty', 'results_backup_TEQL_penalty'),  # With penalty
        ('cartpole', 'convergent', 'results_backup_TEQL_wo_penalty'),
        ('cartpole', 'convergent_penalty', 'results_backup_TEQL_penalty'),
    ]
    
    for env_name, config_type, results_dir in configs:
        print(f"\n{'='*60}")
        print(f"Running {env_name} - {config_type}")
        print(f"{'='*60}")

        os.makedirs(results_dir, exist_ok=True)
        
        env = CustomPendulumEnv() if env_name == 'pendulum' else CustomContinuousCartPoleEnv()
        
        # create configuration
        if config_type == 'convergent':
            # Without penalty - set lambda_penalty = 0
            config_file = f'parameters/{env_name}_convergent_tlr_learning.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            config['lambda_penalty'] = 0.0
            config['epsilon_penalty'] = 1e-6
        else:  # convergent_penalty
            # With penalty - set lambda_penalty = 0.01
            config_file = f'parameters/{env_name}_convergent_tlr_learning.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            config['lambda_penalty'] = 0.01
            config['epsilon_penalty'] = 1e-6
        
        # create temporary config file - SEPARATE filename from path
        temp_config_filename = f'{env_name}_{config_type}_temp.json'  # Just the filename
        temp_config_path = f'parameters/{temp_config_filename}'       # Full path for writing
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # run the experiments N_ITERATIONS times
        for iteration in range(N_ITERATIONS):
            print(f"Iteration {iteration + 1}/{N_ITERATIONS}")
            
            # run experiment - pass only the filename
            experiment = Experiment(temp_config_filename, env, run_freq=10)  # Only filename!
            experiment.run_experiments(window=30 if env_name == 'pendulum' else 50)
            
            # backup results - keep original project directory structure
            backup_dir = f'{results_dir}/iteration_{iteration}'
            os.makedirs(backup_dir, exist_ok=True)
            
            # The results file will be in results/ directory with the same name as config
            result_file = f'results/{temp_config_filename}'
            if os.path.exists(result_file):
                target_file = os.path.join(backup_dir, f'{env_name}_convergent_tlr_learning.json')
                shutil.move(result_file, target_file)
        
        # clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    print("\nAblation study completed!")

def visualize_results():
    print("\nGenerating visualizations...")
    
    # directly import and call visualization functions
    from visualization.plot_ablation_study import main as plot_ablation
    from visualization.performance_comparison import plot_final_performance_comparison
    from visualization.convergence_analysis import plot_convergence_comparison_main
    
    # run visualizations
    plot_ablation()
    plot_final_performance_comparison()
    stats = plot_convergence_comparison_main()

    print("Visualizations completed!")
    return stats

def main():
    print("TEQL Ablation Study")
    print("Comparison: with penalty vs without penalty")
    print("-"*80)
    
    run_ablation_experiments()
    
    visualize_results()
    
    print("Results saved in:")
    print("  - results_backup_TEQL_wo_penalty/")
    print("  - results_backup_TEQL_penalty/")
    print("Figures saved in: figures/")

if __name__ == "__main__":
    main()