"""
Ablation Study: Effect of Regularization Parameter λ
=====================================================
Generates learning curves and box plots comparing TEQL with λ=0 vs λ>0.

Environments: Pendulum, CartPole, Highway

Usage:
    python ablation.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter1d

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# ============================================================================
# Configuration
# ============================================================================

# Color scheme
COLORS = {
    'teql_lambda_0': '#009E73',   # Green for λ=0
    'teql_lambda_pos': '#D2691E', # Orange/Brown for λ>0
    'tlr': '#F39C12',             # Golden orange for TLR
}

# Data directories
DATA_DIRS = {
    'lambda_0': 'results_backup_TEQL_wo_penalty',  # TEQL with λ=0
    'lambda_pos': 'results_backup_comparison',     # TLR & TEQL with λ>0
}

# Environment configuration
ENV_CONFIG = {
    'pendulum': {
        'target_episodes': 10000,
        'record_freq': 10,
        'final_points': 200,  # Last 2000 episodes
    },
    'cartpole': {
        'target_episodes': 10000,
        'record_freq': 10,
        'final_points': 200,
    },
    'highway': {
        'target_episodes': 10000,
        'record_freq': 10,
        'final_points': 200,
    },
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def smooth_anomalies(data, window_size=5):
    """Smooth anomalies using local window mean replacement."""
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
        if abs(data[i] - window_mean) > 0.3 * abs(window_mean):
            smoothed_data[i] = window_mean
    return smoothed_data


def load_json_data(base_dir, env, algo_type='convergent'):
    """
    Load data for a specific environment and algorithm.
    
    Args:
        base_dir: Base directory containing iteration folders
        env: Environment name ('cartpole', 'pendulum', 'highway')
        algo_type: Algorithm type ('original', 'convergent')
    
    Returns:
        dict with 'greedy_cumulative_reward' list of arrays
    """
    data = {'greedy_cumulative_reward': []}
    
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return data
    
    try:
        iteration_dirs = sorted(
            [d for d in os.listdir(base_dir) if d.startswith('iteration_')],
            key=lambda x: int(x.split('_')[1])
        )
    except Exception as e:
        print(f"Warning: Error listing {base_dir}: {e}")
        return data
    
    loaded_count = 0
    for iteration_dir in iteration_dirs:
        file_name = f'{env}_{algo_type}_tlr_learning.json'
        file_path = os.path.join(base_dir, iteration_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            rewards = json_data.get('greedy_cumulative_reward', [])
            
            # Handle nested list format
            if len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            rewards = np.array(rewards, dtype=np.float64)
            
            if len(rewards) == 0:
                continue
            
            # Smooth anomalies
            rewards = smooth_anomalies(rewards)
            data['greedy_cumulative_reward'].append(rewards)
            loaded_count += 1
            
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            continue
    
    if loaded_count > 0:
        print(f"  Loaded {loaded_count} iterations from {base_dir}/{env}")
    
    return data


def process_data(data_list, max_len=None):
    """Process data list, compute mean and std."""
    if not data_list:
        return np.array([]), np.array([]), 0
    
    min_len = min(len(d) for d in data_list)
    if max_len is not None:
        min_len = min(min_len, max_len)
    
    trimmed = [d[:min_len] for d in data_list]
    arr = np.array(trimmed)
    return np.mean(arr, axis=0), np.std(arr, axis=0), min_len


def apply_post_smoothing(data, window=20):
    """Apply smoothing filter to averaged data."""
    if len(data) < window:
        return data
    smoothed = uniform_filter1d(data, size=window)
    return uniform_filter1d(smoothed, size=window // 2)


def get_final_performance(reward_data, final_points=200):
    """Calculate average reward of last N data points for each run."""
    final_performances = []
    for run_rewards in reward_data:
        if len(run_rewards) == 0:
            continue
        n_points = min(final_points, len(run_rewards))
        final_performances.append(np.mean(run_rewards[-n_points:]))
    return final_performances


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_learning_curves():
    """Plot learning curves for λ=0 vs λ>0 across all environments."""
    print("\n" + "=" * 70)
    print("Generating Learning Curves: TEQL (λ=0) vs TEQL (λ>0)")
    print("=" * 70)
    
    environments = ['pendulum', 'cartpole', 'highway']
    smooth_window = 50
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
    
    for idx, env in enumerate(environments):
        ax = axes[idx]
        config = ENV_CONFIG[env]
        max_points = config['target_episodes'] // config['record_freq']
        
        print(f"\nProcessing {env.capitalize()}...")
        
        # Load data
        data_lambda_0 = load_json_data(DATA_DIRS['lambda_0'], env, 'convergent')
        data_lambda_pos = load_json_data(DATA_DIRS['lambda_pos'], env, 'convergent')
        
        # Process λ=0 data
        mean_0, std_0, len_0 = process_data(
            data_lambda_0['greedy_cumulative_reward'], max_len=max_points
        )
        x_0 = np.arange(0, len_0 * config['record_freq'], config['record_freq'])
        
        # Process λ>0 data (truncate to match λ=0 length)
        mean_pos, std_pos, len_pos = process_data(
            data_lambda_pos['greedy_cumulative_reward'], max_len=len_0
        )
        x_pos = np.arange(0, len_pos * config['record_freq'], config['record_freq'])
        
        # Apply smoothing
        if len_0 > 0:
            mean_0_smooth = apply_post_smoothing(mean_0, smooth_window)
            std_0_smooth = apply_post_smoothing(std_0, smooth_window)
        if len_pos > 0:
            mean_pos_smooth = apply_post_smoothing(mean_pos, smooth_window)
            std_pos_smooth = apply_post_smoothing(std_pos, smooth_window)
        
        # Plot λ=0 (dashed green)
        if len_0 > 0:
            ax.plot(x_0, mean_0_smooth, label=r'TEQL ($\lambda$=0)', 
                    color=COLORS['teql_lambda_0'], linewidth=2.5, linestyle='--')
            ax.fill_between(x_0, mean_0_smooth - std_0_smooth, mean_0_smooth + std_0_smooth,
                           alpha=0.15, color=COLORS['teql_lambda_0'])
        
        # Plot λ>0 (solid orange)
        if len_pos > 0:
            ax.plot(x_pos, mean_pos_smooth, label=r'TEQL ($\lambda$>0)',
                    color=COLORS['teql_lambda_pos'], linewidth=2.5, linestyle='-')
            ax.fill_between(x_pos, mean_pos_smooth - std_pos_smooth, mean_pos_smooth + std_pos_smooth,
                           alpha=0.15, color=COLORS['teql_lambda_pos'])
        
        ax.set_title(env.capitalize(), fontweight='bold', fontsize=20)
        ax.set_xlabel('Episodes', fontweight='bold')
        ax.set_ylabel('Cumulative Reward', fontweight='bold')
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/ablation_curves.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig('figures/ablation_curves.png', dpi=300, bbox_inches='tight', format='png')
    print("\n✓ Saved: figures/ablation_curves.pdf")
    plt.show()


def plot_final_performance_boxplot():
    """Plot box plots of final performance for all environments."""
    print("\n" + "=" * 70)
    print("Generating Box Plots: Final Performance Distribution")
    print("=" * 70)
    
    environments = ['pendulum', 'cartpole', 'highway']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.18, wspace=0.28)
    
    # Colors: TLR (orange), λ=0 (green), λ>0 (red/brown)
    colors = [COLORS['tlr'], COLORS['teql_lambda_0'], COLORS['teql_lambda_pos']]
    
    all_statistics = {}
    
    for idx, env in enumerate(environments):
        ax = axes[idx]
        config = ENV_CONFIG[env]
        max_points = config['target_episodes'] // config['record_freq']
        
        print(f"\nProcessing {env.capitalize()}...")
        
        # Load TLR data (from lambda_pos directory)
        tlr_data = load_json_data(DATA_DIRS['lambda_pos'], env, 'original')
        # Load TEQL λ=0 data
        teql_0_data = load_json_data(DATA_DIRS['lambda_0'], env, 'convergent')
        # Load TEQL λ>0 data
        teql_pos_data = load_json_data(DATA_DIRS['lambda_pos'], env, 'convergent')
        
        # Truncate to target episodes
        tlr_rewards = [r[:max_points] for r in tlr_data['greedy_cumulative_reward']]
        teql_0_rewards = [r[:max_points] for r in teql_0_data['greedy_cumulative_reward']]
        teql_pos_rewards = [r[:max_points] for r in teql_pos_data['greedy_cumulative_reward']]
        
        # Calculate final performance
        tlr_final = get_final_performance(tlr_rewards, config['final_points'])
        teql_0_final = get_final_performance(teql_0_rewards, config['final_points'])
        teql_pos_final = get_final_performance(teql_pos_rewards, config['final_points'])
        
        print(f"  TLR: {len(tlr_final)} runs")
        print(f"  TEQL (λ=0): {len(teql_0_final)} runs")
        print(f"  TEQL (λ>0): {len(teql_pos_final)} runs")
        
        # Prepare box plot
        data_to_plot = [tlr_final, teql_0_final, teql_pos_final]
        labels = ['TLR', r'TEQL ($\lambda$=0)', r'TEQL ($\lambda$>0)']
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                              notch=True, showmeans=True, meanline=True,
                              showfliers=True, widths=0.6)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for flier in box_plot['fliers']:
            flier.set(marker='o', markerfacecolor='red', markersize=6, alpha=0.6)
        
        ax.set_title(env.capitalize(), fontweight='bold', fontsize=18)
        ax.set_ylabel('Final Average Rewards', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
        
        # Set y-axis range
        all_finals = tlr_final + teql_0_final + teql_pos_final
        if all_finals:
            min_val, max_val = min(all_finals), max(all_finals)
            margin = (max_val - min_val) * 0.1
            ax.set_ylim(min_val - margin, max_val + margin)
        
        # Store statistics
        all_statistics[env] = {
            'TLR': compute_stats(tlr_final),
            'TEQL (λ=0)': compute_stats(teql_0_final),
            'TEQL (λ>0)': compute_stats(teql_pos_final),
        }
    
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/ablation_boxplot.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig('figures/ablation_boxplot.png', dpi=300, bbox_inches='tight', format='png')
    print("\n✓ Saved: figures/ablation_boxplot.pdf")
    plt.show()
    
    return all_statistics


def compute_stats(data):
    """Compute statistics for a data list."""
    if not data:
        return {'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0, 'n': 0}
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'n': len(data)
    }


# ============================================================================
# Statistical Tables
# ============================================================================

def print_statistics(stats):
    """Print statistics to console."""
    print("\n" + "=" * 100)
    print("TEQL REGULARIZATION ABLATION STUDY - FINAL PERFORMANCE STATISTICS")
    print("=" * 100)
    print("\nNote: Final performance is calculated as the average reward of the last 2000 episodes")
    print("      (200 data points with recording frequency of 10 episodes)\n")
    
    for env, env_stats in stats.items():
        print("-" * 80)
        print(f"{env.upper()} ENVIRONMENT")
        print("-" * 80)
        print(f"{'Method':<20} {'Mean':>12} {'Std':>12} {'Median':>12} {'Min':>12} {'Max':>12} {'N':>8}")
        print("-" * 92)
        for method, values in env_stats.items():
            print(f"{method:<20} {values['mean']:>12.2f} {values['std']:>12.2f} "
                  f"{values['median']:>12.2f} {values['min']:>12.2f} {values['max']:>12.2f} {values['n']:>8}")
        print()


def save_statistics_to_txt(stats, filepath='figures/ablation_statistics.txt'):
    """Save statistics to a txt file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("TEQL REGULARIZATION ABLATION STUDY - FINAL PERFORMANCE STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        f.write("Note: Final performance is calculated as the average reward of the last 2000 episodes\n")
        f.write("      (200 data points with recording frequency of 10 episodes)\n\n")
        
        for env, env_stats in stats.items():
            f.write("-" * 80 + "\n")
            f.write(f"{env.upper()} ENVIRONMENT\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"{'Method':<20} {'Mean':>12} {'Std':>12} {'Median':>12} {'Min':>12} {'Max':>12} {'N':>8}\n")
            f.write("-" * 92 + "\n")
            for method, values in env_stats.items():
                f.write(f"{method:<20} {values['mean']:>12.2f} {values['std']:>12.2f} "
                       f"{values['median']:>12.2f} {values['min']:>12.2f} {values['max']:>12.2f} {values['n']:>8}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF STATISTICS\n")
        f.write("=" * 100 + "\n")
    
    print(f"\n✓ Statistics saved to: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function."""
    print("=" * 80)
    print("TEQL Regularization Ablation Study")
    print("=" * 80)
    print("\nEnvironments: Pendulum, CartPole, Highway")
    print("Comparison: TEQL (λ=0) vs TEQL (λ>0)")
    print("\nData Sources:")
    print(f"  - TEQL (λ=0):  {DATA_DIRS['lambda_0']}")
    print(f"  - TEQL (λ>0):  {DATA_DIRS['lambda_pos']}")
    print(f"  - TLR:         {DATA_DIRS['lambda_pos']}")
    
    # Generate learning curves
    print("\n" + "#" * 80)
    print("# Part 1: Learning Curves")
    print("#" * 80)
    plot_learning_curves()
    
    # Generate box plots and statistics
    print("\n" + "#" * 80)
    print("# Part 2: Final Performance Box Plots")
    print("#" * 80)
    statistics = plot_final_performance_boxplot()
    
    # Print and save statistics
    print_statistics(statistics)
    save_statistics_to_txt(statistics)
    
    print("\n" + "=" * 80)
    print("Analysis Completed!")
    print("Outputs saved in 'figures/' folder:")
    print("  - ablation_curves.pdf (learning curves)")
    print("  - ablation_boxplot.pdf (box plots)")
    print("  - ablation_statistics.txt (statistics table)")
    print("=" * 80)


if __name__ == "__main__":
    main()
