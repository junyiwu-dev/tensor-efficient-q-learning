import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# Set font and style for English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_json_data(base_dir, env, mode):
    """
    Load JSON data from the specified directory - supports iteration_X/ structure
    """
    data = {
        'greedy_steps': [],
        'greedy_cumulative_reward': []
    }
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist!")
        return data
    
    print(f"Processing directory: {base_dir}")
    
    # Find all iteration directories
    iteration_dirs = [d for d in os.listdir(base_dir) 
                     if d.startswith('iteration_') and os.path.isdir(os.path.join(base_dir, d))]
    
    if not iteration_dirs:
        print(f"No iteration directories found in {base_dir}")
        return data
    
    # Sort by iteration number
    iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
    print(f"Found {len(iteration_dirs)} iteration directories")
    
    # Build target filename
    target_filename = f"{env}_{mode}_tlr_learning.json"
    
    for iteration_dir in iteration_dirs:
        iteration_path = os.path.join(base_dir, iteration_dir)
        file_path = os.path.join(iteration_path, target_filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist, skipping...")
            continue
            
        print(f"Loading: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                
            # Extract greedy_steps
            if 'greedy_steps' in file_data:
                steps = file_data['greedy_steps']
                if isinstance(steps, list) and len(steps) > 0:
                    data['greedy_steps'].append(steps)
                    
            # Extract greedy_cumulative_reward
            if 'greedy_cumulative_reward' in file_data:
                rewards = file_data['greedy_cumulative_reward']
                
                # Handle nested list case
                if len(rewards) > 0 and isinstance(rewards[0], list):
                    rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                
                rewards = np.array(rewards)
                if len(rewards) == 0:
                    print(f"Warning: Empty greedy_cumulative_reward in {file_path}, skipping...")
                    continue
                    
                data['greedy_cumulative_reward'].append(rewards)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(data['greedy_steps'])} greedy_steps and {len(data['greedy_cumulative_reward'])} greedy_cumulative_reward")
    
    return data

def process_data(data_list):
    """
    Process data, calculate mean and standard deviation, ensure consistent length
    """
    if not data_list:
        print("Warning: No data to process")
        return np.array([]), np.array([]), 0
        
    min_len = min(len(data) for data in data_list)
    trimmed_data = [data[:min_len] for data in data_list]
    data_array = np.array(trimmed_data)
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    
    return mean_data, std_data, min_len

def apply_post_smoothing(data, smooth_window=50):
    """
    Apply smoothing to averaged data using moving average filter
    """
    if len(data) < smooth_window:
        print(f"Warning: Data length {len(data)} is too short for window_size {smooth_window}, using length/2")
        smooth_window = max(1, len(data) // 2)
        
    smoothed = uniform_filter1d(data, size=smooth_window)
    return smoothed

def plot_discretization_sensitivity_combined(algorithm, config_names, config_dirs, 
                                           metric='greedy_cumulative_reward', smooth_window=50):
    """
    Plot discretization sensitivity analysis - left and right environments
    """
    # Define color scheme - light to dark represents coarse to fine discretization
    colors = ['#FFB6C1', '#FF69B4', '#DC143C', '#B22222', '#8B0000']  # Pink to dark red gradient
    linestyles = ['-', '-', '-', '-', '-']  # All solid lines
    linewidths = [2.5, 2.5, 3.0, 2.5, 2.5]  # Current setting slightly thicker
    
    environments = ['pendulum', 'cartpole']
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for env_idx, env_name in enumerate(environments):
        ax = axes[env_idx]
        
        for i, (config_name, config_dir) in enumerate(zip(config_names, config_dirs)):
            print(f"\n=== Processing {config_name} for {env_name} {metric} ===")
            
            # Select mode based on algorithm
            mode = 'original' if algorithm == 'TLR' else 'convergent'
            data = load_json_data(config_dir, env_name, mode)
            
            # Process data
            mean_data, std_data, data_len = process_data(data[metric])
            
            if data_len > 0:
                # Apply smoothing
                mean_smooth = apply_post_smoothing(mean_data, smooth_window)
                std_smooth = apply_post_smoothing(std_data, smooth_window)
                
                x = np.arange(0, data_len * 10, 10)
                
                # Plot curve
                ax.plot(x, mean_smooth, label=config_name.replace('_', ' ').title(), 
                       color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i])
                ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, 
                               alpha=0.2, color=colors[i], edgecolor='none')
            else:
                print(f"Warning: No data found for {config_name}")
        
        # Set axis properties
        ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=16)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=16)
        ax.set_title(f'{env_name.capitalize()}', fontweight='bold', fontsize=18)
        
        # Set legend - only show on first subplot
        if env_idx == 0:
            ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', fontsize=12)
        
        # Set grid and style
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function: generates two PDF files, each containing left and right environments
    """
    # Discretization configurations
    config_names = ['very_coarse', 'coarse', 'current', 'fine', 'very_fine']
    config_dirs = ['results_backup_very_coarse', 'results_backup_coarse', 
                   'results_backup_current', 'results_backup_fine', 'results_backup_very_fine']
    
    smooth_window = 50
    
    # Create figures folder
    os.makedirs('figures', exist_ok=True)
    
    # 1. Generate TLR discretization sensitivity analysis (Greedy Cumulative Reward)
    print("="*80)
    print("GENERATING TLR DISCRETIZATION SENSITIVITY ANALYSIS")
    print("="*80)
    
    fig = plot_discretization_sensitivity_combined('TLR', config_names, config_dirs, 
                                                  metric='greedy_cumulative_reward', 
                                                  smooth_window=smooth_window)
    
    # Add overall title
    fig.suptitle('TLR Discretization Sensitivity Analysis - Greedy Cumulative Reward', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Save PDF
    pdf_filename = 'figures/TLR_discretization_sensitivity.pdf'
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {pdf_filename}")
    
    plt.show()
    
    # 2. Generate TEQL discretization sensitivity analysis (Greedy Cumulative Reward)
    print("\n" + "="*80)
    print("GENERATING TEQL DISCRETIZATION SENSITIVITY ANALYSIS")
    print("="*80)
    
    fig = plot_discretization_sensitivity_combined('TEQL', config_names, config_dirs, 
                                                  metric='greedy_cumulative_reward', 
                                                  smooth_window=smooth_window)
    
    # Add overall title
    fig.suptitle('TEQL Discretization Sensitivity Analysis - Greedy Cumulative Reward', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Save PDF
    pdf_filename = 'figures/TEQL_discretization_sensitivity.pdf'
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {pdf_filename}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("DISCRETIZATION SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print("Generated files:")
    print("  - figures/TLR_discretization_sensitivity.pdf")
    print("  - figures/TEQL_discretization_sensitivity.pdf")

if __name__ == "__main__":
    main()