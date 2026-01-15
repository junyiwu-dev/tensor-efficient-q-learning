"""
Plot Multiple Environments: Multi-Algorithm Comparison
=======================================================
Generate publication-quality figures comparing algorithms across environments.

Algorithms: TLR, TEQL, LoRa-VI, DQN, SAC
Environments: CartPole, Pendulum, Highway

Usage:
    # Plot all 5 algorithms on 2 environments
    python plot_baselines.py
    
    # Plot specific algorithms
    python plot_baselines.py --model teql dqn sac
    
    # Plot specific environments
    python plot_baselines.py --env cartpole pendulum --model sac dqn lora-vi tlr teql
    
    # Plot only highway
    python plot_baselines.py --env highway --model dqn sac tlr teql
    
    # Specify custom directories
    python plot_baselines.py --teql-dir results_backup --dqn-dir results_backup
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # For smoothing

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'axes.labelsize': 26,
    'axes.titlesize': 30,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'lines.linewidth': 3.0,
    'figure.figsize': (16, 10),
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
})

# ============================================================================
# Algorithm Configuration
# ============================================================================

# Algorithm display settings
# Color scheme: Tensor methods (warm tones) vs Deep RL (cool/neutral tones)
ALGO_CONFIG = {
    'teql': {
        'label': 'TEQL',
        'color': '#C0392B',   
        'linestyle': '-',
        'file_pattern': '{env}_convergent_tlr_learning.json',
        'default_dir': 'results_backup'
    },
    'tlr': {
        'label': 'TLR',
        'color': '#F39C12',      
        'linestyle': '--',
        'file_pattern': '{env}_original_tlr_learning.json',
        'default_dir': 'results_backup'
    },
    'lora-vi': {
        'label': 'LoRa-VI',
        'color': '#27AE60',     
        'linestyle': '-.',
        'file_pattern': '{env}_lora_vi_learning.json',
        'default_dir': 'results_backup'
    },
    'dqn': {
        'label': 'DQN',
        'color': '#3498DB',
        'linestyle': ':',
        'file_pattern': '{env}_dqn.json',
        'default_dir': 'results_backup'
    },
    'sac': {
        'label': 'SAC',
        'color': '#95A5A6',      
        'linestyle': '-',
        'file_pattern': '{env}_sac.json',
        'default_dir': 'results_backup'
    }
}

# Default algorithms to plot
DEFAULT_ALGORITHMS = ['tlr', 'teql', 'lora-vi', 'dqn', 'sac']

# Default environments
DEFAULT_ENVIRONMENTS = ['cartpole', 'pendulum']

# All available environments
ALL_ENVIRONMENTS = ['cartpole', 'pendulum', 'highway']


# ============================================================================
# Data Loading Functions
# ============================================================================

def smooth_anomalies(data, window_size=5):
    """Smooth anomalies in data"""
    if len(data) < 2 * window_size + 1:
        return data.copy()
        
    smoothed_data = data.copy()
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        
        window_values = data[start:i].tolist() + data[i+1:end].tolist()
        if not window_values:
            continue
        
        window_mean = np.mean(window_values)
        
        # If the current point differs too much from the window mean
        if abs(data[i] - window_mean) > 0.3 * abs(window_mean) if window_mean != 0 else False:
            smoothed_data[i] = window_mean
    
    return smoothed_data


def load_algorithm_data(base_dir, env, file_pattern):
    """
    Load data from JSON files in backup directory structure.
    
    Args:
        base_dir: Base directory containing iteration_X folders
        env: Environment name ('cartpole', 'pendulum', or 'highway')
        file_pattern: File name pattern with {env} placeholder
        
    Returns:
        dict with 'greedy_steps' and 'greedy_cumulative_reward' lists
    """
    data = {'greedy_steps': [], 'greedy_cumulative_reward': []}
    
    if not os.path.exists(base_dir):
        print(f"  Warning: Directory '{base_dir}' does not exist")
        return data
    
    # Get sorted iteration directories
    try:
        iteration_dirs = sorted(
            [d for d in os.listdir(base_dir) if d.startswith('iteration_')],
            key=lambda x: int(x.split('_')[1])
        )
    except Exception as e:
        print(f"  Warning: Error listing directory '{base_dir}': {e}")
        return data
    
    if not iteration_dirs:
        print(f"  Warning: No iteration_X directories found in '{base_dir}'")
        return data
    
    loaded_count = 0
    for iteration_dir in iteration_dirs:
        file_name = file_pattern.format(env=env)
        file_path = os.path.join(base_dir, iteration_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Process greedy_steps
            steps = json_data.get('greedy_steps', [])
            # Handle nested list format (pendulum and highway may have this)
            if env in ['pendulum', 'highway'] and len(steps) > 0 and isinstance(steps[0], list):
                steps = [s[0] if isinstance(s, list) else s for s in steps]
            
            steps = np.array(steps, dtype=np.float64)
            if len(steps) == 0:
                continue
            
            steps = smooth_anomalies(steps)
            data['greedy_steps'].append(steps)
            
            # Process greedy_cumulative_reward
            rewards = json_data.get('greedy_cumulative_reward', [])
            # Handle nested list format (pendulum and highway may have this)
            if env in ['pendulum', 'highway'] and len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            rewards = np.array(rewards, dtype=np.float64)
            if len(rewards) == 0:
                continue
            
            rewards = smooth_anomalies(rewards)
            data['greedy_cumulative_reward'].append(rewards)
            
            loaded_count += 1
            
        except Exception as e:
            print(f"  Warning: Error loading {file_path}: {e}")
            continue
    
    if loaded_count > 0:
        print(f"  ✓ Loaded {loaded_count} iterations")
    
    return data


def process_data(data_list):
    """Process multiple runs to get mean and std"""
    if not data_list:
        return np.array([]), np.array([]), 0
        
    min_len = min(len(data) for data in data_list)
    trimmed_data = [data[:min_len] for data in data_list]
    data_array = np.array(trimmed_data)
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    
    return mean_data, std_data, min_len


def apply_post_smoothing(data, smooth_window=20):
    """Apply post-processing smoothing"""
    if len(data) < smooth_window:
        return data
        
    # Apply filter twice for smoother results
    smoothed_once = uniform_filter1d(data, size=smooth_window)
    smoothed_twice = uniform_filter1d(smoothed_once, size=smooth_window//2)
    
    return smoothed_twice


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_algorithms(ax, algo_data_dict, metric, title, smooth_window=20):
    """
    Plot comparison of multiple algorithms.
    
    Args:
        ax: Matplotlib axis
        algo_data_dict: Dict of {algo_name: data_dict}
        metric: Metric to plot ('greedy_cumulative_reward' or 'greedy_steps')
        title: Plot title
        smooth_window: Smoothing window size
    """
    for algo_name, data in algo_data_dict.items():
        if algo_name not in ALGO_CONFIG:
            print(f"  Warning: Unknown algorithm '{algo_name}', skipping")
            continue
            
        config = ALGO_CONFIG[algo_name]
        
        # Process data
        mean_data, std_data, data_len = process_data(data[metric])
        
        if data_len == 0:
            continue
            
        x = np.arange(0, data_len * 10, 10)
        
        # Apply post-smoothing
        mean_smooth = apply_post_smoothing(mean_data, smooth_window)
        std_smooth = apply_post_smoothing(std_data, smooth_window)
        
        # Plot mean line
        ax.plot(x, mean_smooth, 
                label=config['label'], 
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=3.0)
        
        # Plot confidence band
        ax.fill_between(x, 
                       mean_smooth - 0.5*std_smooth, 
                       mean_smooth + 0.5*std_smooth,
                       alpha=0.12, 
                       color=config['color'], 
                       edgecolor='none')
    
    ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=24)
    ax.set_ylabel('Cumulative Reward', fontweight='bold', fontsize=24)
    ax.set_title(title, fontweight='bold', fontsize=26)
    
    # Set legend
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, 
             facecolor='white', fontsize=18)
    
    # Set grid style
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Axis styling
    ax.tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate multi-algorithm comparison plots"""
    
    parser = argparse.ArgumentParser(
        description='Plot algorithm comparison across environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all 5 algorithms (default)
  python plot_baselines.py
  
  # Plot only specific algorithms
  python plot_baselines.py --model teql dqn sac
  python plot_baselines.py --model tlr teql
  
  # Plot specific environments
  python plot_baselines.py --env cartpole pendulum highway
  python plot_baselines.py --env highway --model teql dqn sac
  
  # Specify custom directories for each algorithm
  python plot_baselines.py --teql-dir results_backup --dqn-dir results_backup_dqn
  
  # Use a single directory for all algorithms
  python plot_baselines.py --base-dir results_backup

  # Specify output filename
  python plot_baselines.py --output my_comparison

Available algorithms: tlr, teql, lora-vi, dqn, sac
Available environments: cartpole, pendulum, highway
        """
    )
    
    parser.add_argument('--model', type=str, nargs='+', 
                        choices=['tlr', 'teql', 'lora-vi', 'dqn', 'sac'],
                        default=None,
                        help='Algorithms to plot (default: all)')
    parser.add_argument('--env', type=str, nargs='+',
                        choices=['cartpole', 'pendulum', 'highway'],
                        default=DEFAULT_ENVIRONMENTS,
                        help='Environments to plot (default: cartpole, pendulum)')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for all algorithms (overrides individual dirs)')
    parser.add_argument('--tlr-dir', type=str, default='results_backup',
                        help='Directory for TLR results')
    parser.add_argument('--teql-dir', type=str, default='results_backup',
                        help='Directory for TEQL results')
    parser.add_argument('--lora-vi-dir', type=str, default='results_backup',
                        help='Directory for LoRa-VI results')
    parser.add_argument('--dqn-dir', type=str, default='results_backup',
                        help='Directory for DQN results')
    parser.add_argument('--sac-dir', type=str, default='results_backup',
                        help='Directory for SAC results')
    parser.add_argument('--smooth', type=int, default=30,
                        help='Smoothing window size')
    parser.add_argument('--output', type=str, default='comparison_all_algorithms',
                        help='Output filename (without extension)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (only save)')
    
    args = parser.parse_args()
    
    # Determine which algorithms to plot
    algorithms = args.model if args.model else DEFAULT_ALGORITHMS
    environments = args.env
    smooth_window = args.smooth
    
    # Build directory mapping
    if args.base_dir:
        # Use base_dir for all algorithms
        algo_dirs = {algo: args.base_dir for algo in algorithms}
    else:
        # Use individual directories
        algo_dirs = {
            'tlr': args.tlr_dir,
            'teql': args.teql_dir,
            'lora-vi': args.lora_vi_dir,
            'dqn': args.dqn_dir,
            'sac': args.sac_dir
        }
    
    print("=" * 70)
    print("Multi-Algorithm Comparison Across Environments")
    print("=" * 70)
    print(f"Algorithms: {', '.join([ALGO_CONFIG[a]['label'] for a in algorithms])}")
    print(f"Environments: {', '.join([e.capitalize() for e in environments])}")
    print(f"Smoothing window: {smooth_window}")
    print()
    
    # Print directory configuration
    print("Directory configuration:")
    for algo in algorithms:
        print(f"  {ALGO_CONFIG[algo]['label']}: {algo_dirs.get(algo, 'N/A')}")
    print()
    
    # Create figure
    n_envs = len(environments)
    fig, axes = plt.subplots(1, n_envs, figsize=(9 * n_envs, 7))
    
    if n_envs == 1:
        axes = [axes]
    
    # Process each environment
    for idx, env in enumerate(environments):
        print(f"=== Processing {env.upper()} ===")
        
        # Load data for each algorithm
        algo_data = {}
        for algo in algorithms:
            base_dir = algo_dirs.get(algo, ALGO_CONFIG[algo]['default_dir'])
            file_pattern = ALGO_CONFIG[algo]['file_pattern']
            
            print(f"Loading {ALGO_CONFIG[algo]['label']} from {base_dir}...")
            data = load_algorithm_data(base_dir, env, file_pattern)
            
            if data['greedy_cumulative_reward']:
                algo_data[algo] = data
                print(f"  ✓ {ALGO_CONFIG[algo]['label']}: {len(data['greedy_cumulative_reward'])} runs")
            else:
                print(f"  ✗ {ALGO_CONFIG[algo]['label']}: No data found")
        
        if not algo_data:
            print(f"Warning: No data loaded for {env}! Skipping.")
            continue
        
        # Plot
        plot_algorithms(axes[idx], algo_data, 'greedy_cumulative_reward', 
                       env.capitalize(), smooth_window)
        print()
    
    plt.tight_layout()
    
    # Create figures folder
    os.makedirs('figures', exist_ok=True)
    
    # Generate output filename based on algorithms
    if args.model:
        algo_suffix = '_'.join(args.model)
        output_base = f'figures/{args.output}_{algo_suffix}'
    else:
        output_base = f'figures/{args.output}'
    
    # Save as PDF
    output_pdf = f'{output_base}.pdf'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {output_pdf}")
    
    # Save as PNG
    output_png = f'{output_base}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ Saved: {output_png}")
    
    if not args.no_show:
        plt.show()
    
    print("\n" + "=" * 70)
    print("Plotting complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()