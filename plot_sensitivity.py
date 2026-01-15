"""
Unified Sensitivity Analysis Plotting (TEQL & TLR)
===================================================
Generate publication-quality figures showing how discretization
affects TEQL and TLR learning performance.

Modes:
- teql-only: Plot TEQL performance across discretization levels (5 levels)
- tlr-vs-teql: Plot TLR vs TEQL comparison (4 levels, 2x4 grid)

Usage:
    # Plot TEQL-only sensitivity analysis (default)
    python plot_sensitivity.py --mode teql-only
    
    # Plot TLR vs TEQL comparison (2x4 grid)
    python plot_sensitivity.py --mode tlr-vs-teql
    
    # Plot specific environment (TEQL-only mode)
    python plot_sensitivity.py --mode teql-only --env cartpole

    # Plot specific levels (TEQL-only mode)
    python plot_sensitivity.py --mode teql-only --level fine coarse
    
    # Custom smoothing
    python plot_sensitivity.py --mode tlr-vs-teql --smooth 30
    
    # Print summary tables only
    python plot_sensitivity.py --mode teql-only --summary
    python plot_sensitivity.py --mode tlr-vs-teql --summary
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass


# ============================================================================
# Configuration
# ============================================================================

BACKUP_DIR = 'results_backup_sensitivity copy'

# Level order for TEQL-only plotting (5 levels)
LEVEL_ORDER_TEQL = ['very_coarse', 'coarse', 'medium', 'fine', 'very_fine']

# Level order for TLR vs TEQL plotting (4 levels)
LEVEL_ORDER_COMPARISON = ['very_coarse', 'coarse', 'fine', 'very_fine']

# Display names for levels
LEVEL_DISPLAY_NAMES = {
    'very_coarse': 'Very Coarse',
    'coarse': 'Coarse',
    'medium': 'Medium',
    'fine': 'Fine',
    'very_fine': 'Very Fine'
}

# Environment display names
ENV_DISPLAY_NAMES = {
    'pendulum': 'Pendulum',
    'cartpole': 'CartPole'
}

# Level display configuration for TEQL-only mode (5 levels)
LEVEL_CONFIG_TEQL = {
    'very_coarse': {
        'label': 'Very Coarse',
        'color': '#e74c3c',      # Red
        'linestyle': ':',
    },
    'coarse': {
        'label': 'Coarse',
        'color': '#f39c12',      # Orange
        'linestyle': '--',
    },
    'medium': {
        'label': 'Medium',
        'color': '#9b59b6',      # Purple
        'linestyle': '-.',
    },
    'fine': {
        'label': 'Fine',
        'color': '#3498db',      # Blue
        'linestyle': (0, (3, 1, 1, 1)),  # densely dashdotted
    },
    'very_fine': {
        'label': 'Very Fine',
        'color': '#2ecc71',      # Green
        'linestyle': '-',
    }
}

# Algorithm configuration for TLR vs TEQL mode
ALGO_CONFIG = {
    'tlr': {
        'label': 'TLR',
        'color': '#0072B2',     # Blue (colorblind friendly)
        'linestyle': '--',
    },
    'teql': {
        'label': 'TEQL',
        'color': '#D2691E',     # Orange-brown
        'linestyle': '-',
    }
}

# Discretization configurations for TEQL-only mode (5 levels including medium)
DISCRETIZATION_LEVELS_TEQL = {
    'pendulum': {
        'very_fine': {'bucket_states': [80, 80], 'bucket_actions': [40]},
        'fine': {'bucket_states': [30, 30], 'bucket_actions': [15]},
        'medium': {'bucket_states': [20, 20], 'bucket_actions': [10]},
        'coarse': {'bucket_states': [15, 15], 'bucket_actions': [8]},
        'very_coarse': {'bucket_states': [8, 8], 'bucket_actions': [4]}
    },
    'cartpole': {
        'very_fine': {'bucket_states': [40, 40, 80, 80], 'bucket_actions': [40]},
        'fine': {'bucket_states': [15, 15, 30, 30], 'bucket_actions': [15]},
        'medium': {'bucket_states': [10, 10, 20, 20], 'bucket_actions': [10]},
        'coarse': {'bucket_states': [8, 8, 15, 15], 'bucket_actions': [8]},
        'very_coarse': {'bucket_states': [5, 5, 8, 8], 'bucket_actions': [4]}
    }
}

# Discretization configurations for TLR vs TEQL mode (4 levels, no medium)
DISCRETIZATION_LEVELS_COMPARISON = {
    'pendulum': {
        'very_fine': {'bucket_states': [80, 80], 'bucket_actions': [40]},
        'fine': {'bucket_states': [30, 30], 'bucket_actions': [15]},
        'coarse': {'bucket_states': [15, 15], 'bucket_actions': [8]},
        'very_coarse': {'bucket_states': [8, 8], 'bucket_actions': [4]}
    },
    'cartpole': {
        'very_fine': {'bucket_states': [40, 40, 80, 80], 'bucket_actions': [40]},
        'fine': {'bucket_states': [15, 15, 30, 30], 'bucket_actions': [15]},
        'coarse': {'bucket_states': [8, 8, 15, 15], 'bucket_actions': [8]},
        'very_coarse': {'bucket_states': [5, 5, 8, 8], 'bucket_actions': [4]}
    }
}


# ============================================================================
# Plot Style Configuration Functions
# ============================================================================

def set_teql_only_style():
    """Set plot style for TEQL-only mode."""
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'figure.figsize': (14, 6),
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })


def set_comparison_style():
    """Set plot style for TLR vs TEQL comparison mode."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_result_filename(env_name, algo, level):
    """Get result filename for a specific env, algorithm and level."""
    return f"{env_name}_{algo}_{level}.json"


def smooth_data(data, smooth_window=20):
    """Apply post-processing smoothing (two passes for smooth result)"""
    if len(data) < smooth_window:
        return data
    
    # Two-pass smoothing for smooth results
    smoothed_once = uniform_filter1d(data, size=smooth_window)
    smoothed_twice = uniform_filter1d(smoothed_once, size=smooth_window // 2)
    return smoothed_twice


def smooth_anomalies(data, window_size=5):
    """Remove anomalies from data using local smoothing."""
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
        if abs(data[i] - window_mean) > 0.3 * abs(window_mean):
            smoothed_data[i] = window_mean
    
    return smoothed_data


def load_sensitivity_data(env_name, algo, level):
    """Load data for a specific environment, algorithm and discretization level."""
    data = {'greedy_steps': [], 'greedy_cumulative_reward': []}
    
    if not os.path.exists(BACKUP_DIR):
        print(f"Warning: Backup directory '{BACKUP_DIR}' not found")
        return data
    
    # Get iteration directories
    try:
        iteration_dirs = sorted(
            [d for d in os.listdir(BACKUP_DIR) if d.startswith('iteration_')],
            key=lambda x: int(x.split('_')[1])
        )
    except Exception as e:
        print(f"Warning: Error listing directory: {e}")
        return data
    
    loaded_count = 0
    for iteration_dir in iteration_dirs:
        file_name = get_result_filename(env_name, algo, level)
        file_path = os.path.join(BACKUP_DIR, iteration_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Process greedy data
            steps = json_data.get('greedy_steps', [])
            rewards = json_data.get('greedy_cumulative_reward', [])
            
            # Handle nested list format
            if len(steps) > 0 and isinstance(steps[0], list):
                steps = [s[0] if isinstance(s, list) else s for s in steps]
            if len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            steps = np.array(steps, dtype=np.float64)
            rewards = np.array(rewards, dtype=np.float64)
            
            # Apply anomaly smoothing to individual runs
            steps = smooth_anomalies(steps)
            rewards = smooth_anomalies(rewards)
            
            if len(steps) > 0 and len(rewards) > 0:
                data['greedy_steps'].append(steps)
                data['greedy_cumulative_reward'].append(rewards)
                loaded_count += 1
                
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            continue
    
    if loaded_count > 0:
        print(f"  {algo.upper()}/{level}: Loaded {loaded_count} iterations")
    
    return data


def process_data(data_list):
    """Process multiple runs to get mean and std."""
    if not data_list:
        return np.array([]), np.array([]), 0
    
    min_len = min(len(data) for data in data_list)
    trimmed_data = [data[:min_len] for data in data_list]
    data_array = np.array(trimmed_data)
    
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    
    return mean_data, std_data, min_len


def calculate_params(env_name, level, k=10, mode='teql-only'):
    """Calculate parameter count for a level."""
    if mode == 'teql-only':
        config = DISCRETIZATION_LEVELS_TEQL[env_name][level]
    else:
        config = DISCRETIZATION_LEVELS_COMPARISON[env_name][level]
    bucket_states = config['bucket_states']
    bucket_actions = config['bucket_actions']
    dimensions = bucket_states + bucket_actions
    return sum(d * k for d in dimensions)


# ============================================================================
# Check Existing Results Functions
# ============================================================================

def check_existing_results(algo, n_iterations=30):
    """Check if results already exist in results_backup_sensitivity."""
    print("\n" + "=" * 70)
    print(f"Checking for existing {algo.upper()} results...")
    print("=" * 70)
    
    results_summary = {}
    level_order = LEVEL_ORDER_COMPARISON  # Use 4-level order for checking
    
    for env_name in ['pendulum', 'cartpole']:
        results_summary[env_name] = {}
        for level in level_order:
            count = 0
            for i in range(n_iterations):
                iteration_dir = os.path.join(BACKUP_DIR, f'iteration_{i}')
                file_path = os.path.join(iteration_dir, f"{env_name}_{algo}_{level}.json")
                if os.path.exists(file_path):
                    count += 1
            results_summary[env_name][level] = count
            status = "✓" if count > 0 else "✗"
            print(f"  {env_name}/{level} {algo.upper()}: {count} iterations found {status}")
    
    return results_summary


# ============================================================================
# TEQL-Only Plotting Functions
# ============================================================================

def plot_teql_sensitivity_single_env(ax, env_name, levels, metric='greedy_cumulative_reward', 
                                      smooth_window=20):
    """Plot TEQL sensitivity analysis for a single environment."""
    
    for level in levels:
        if level not in LEVEL_CONFIG_TEQL:
            continue
        
        config = LEVEL_CONFIG_TEQL[level]
        data = load_sensitivity_data(env_name, 'teql', level)
        
        if not data[metric]:
            print(f"  {level}: No data found")
            continue
        
        mean_data, std_data, data_len = process_data(data[metric])
        
        if data_len == 0:
            continue
        
        x = np.arange(0, data_len * 10, 10)  # Episodes (every 10th)
        
        # Apply smoothing
        mean_smooth = smooth_data(mean_data, smooth_window)
        std_smooth = smooth_data(std_data, smooth_window)
        
        # Calculate parameters for label
        params = calculate_params(env_name, level, mode='teql-only')
        label = f"{config['label']} ({params:,} params)"
        
        # Plot mean line
        ax.plot(x, mean_smooth,
                label=label,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2.5)
        
        # Plot confidence band
        ax.fill_between(x,
                       mean_smooth - std_smooth,
                       mean_smooth + std_smooth,
                       alpha=0.16,
                       color=config['color'],
                       edgecolor='none')
    
    ax.set_xlabel('Episodes (every 10th)', fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontweight='bold')
    ax.set_title(f'{env_name.capitalize()}', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set consistent y-axis limits (0 to 100 with slight extension)
    ax.set_ylim(-5, 105)


def plot_teql_sensitivity_comparison(envs, levels, smooth_window=20, output_base='sensitivity_analysis'):
    """Generate TEQL sensitivity analysis plots."""
    
    set_teql_only_style()
    
    print("\n" + "=" * 70)
    print("Plotting TEQL Discretization Sensitivity Analysis")
    print("=" * 70)
    
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(8 * n_envs, 6))
    
    if n_envs == 1:
        axes = [axes]
    
    for idx, env_name in enumerate(envs):
        print(f"\nProcessing {env_name.upper()}...")
        plot_teql_sensitivity_single_env(axes[idx], env_name, levels, 
                                          smooth_window=smooth_window)
    
    plt.tight_layout()
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Save plots
    pdf_path = f'figures/{output_base}.pdf'
    png_path = f'figures/{output_base}.png'
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n✓ Saved: {pdf_path}")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ Saved: {png_path}")
    
    return fig


def print_teql_summary_table(envs, levels):
    """Print summary table of parameters and final performance for TEQL."""
    
    print("\n" + "=" * 90)
    print("TEQL SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 90)
    
    for env_name in envs:
        print(f"\n{env_name.upper()}")
        print("-" * 90)
        print(f"{'Level':<15} {'Buckets':<35} {'Params':>10} {'Final Reward':>15} {'Std':>10}")
        print("-" * 90)
        
        for level in levels:
            if level not in DISCRETIZATION_LEVELS_TEQL[env_name]:
                continue
            config = DISCRETIZATION_LEVELS_TEQL[env_name][level]
            params = calculate_params(env_name, level, mode='teql-only')
            
            # Load data
            data = load_sensitivity_data(env_name, 'teql', level)
            
            if data['greedy_cumulative_reward']:
                mean_data, std_data, _ = process_data(data['greedy_cumulative_reward'])
                if len(mean_data) > 0:
                    # Get last 10% average
                    last_n = max(1, len(mean_data) // 10)
                    final_reward = np.mean(mean_data[-last_n:])
                    final_std = np.mean(std_data[-last_n:])
                else:
                    final_reward = 0
                    final_std = 0
            else:
                final_reward = 0
                final_std = 0
            
            bucket_str = f"{config['bucket_states']} + {config['bucket_actions']}"
            print(f"{level:<15} {bucket_str:<35} {params:>10,} {final_reward:>15.2f} {final_std:>10.2f}")
        
        print("-" * 90)
    
    print("\n")


def print_teql_params_only():
    """Print parameter counts for all levels (TEQL mode)."""
    print("\n" + "=" * 80)
    print("TEQL Parameter Count by Discretization Level")
    print("=" * 80)
    
    for env_name in ['pendulum', 'cartpole']:
        print(f"\n{env_name.upper()}")
        print("-" * 80)
        print(f"{'Level':<15} {'Bucket States':<25} {'Bucket Actions':<15} {'Parameters':>12}")
        print("-" * 80)
        
        for level in LEVEL_ORDER_TEQL:
            if level not in DISCRETIZATION_LEVELS_TEQL[env_name]:
                continue
            config = DISCRETIZATION_LEVELS_TEQL[env_name][level]
            params = calculate_params(env_name, level, mode='teql-only')
            
            print(f"{level:<15} {str(config['bucket_states']):<25} {str(config['bucket_actions']):<15} {params:>12,}")
        
        print("-" * 80)
    
    print("\n")


# ============================================================================
# TLR vs TEQL Comparison Plotting Functions
# ============================================================================

def plot_single_comparison(ax, env_name, level, smooth_window=50):
    """Plot TLR vs TEQL comparison for a single env/level combination."""
    
    # Load data for both algorithms
    tlr_data = load_sensitivity_data(env_name, 'tlr', level)
    teql_data = load_sensitivity_data(env_name, 'teql', level)
    
    metric = 'greedy_cumulative_reward'
    has_data = False
    
    # Process and plot TLR data
    if tlr_data[metric]:
        tlr_mean, tlr_std, tlr_len = process_data(tlr_data[metric])
        
        if tlr_len > 0:
            x = np.arange(0, tlr_len * 10, 10)
            
            # Apply smoothing
            tlr_mean_smooth = smooth_data(tlr_mean, smooth_window)
            tlr_std_smooth = smooth_data(tlr_std, smooth_window)
            
            config = ALGO_CONFIG['tlr']
            
            # Plot mean line
            ax.plot(x, tlr_mean_smooth,
                    label=config['label'],
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=2.0)
            
            # Plot confidence band
            ax.fill_between(x,
                           tlr_mean_smooth - tlr_std_smooth,
                           tlr_mean_smooth + tlr_std_smooth,
                           alpha=0.2,
                           color=config['color'],
                           edgecolor='none')
            has_data = True
    
    # Process and plot TEQL data
    if teql_data[metric]:
        teql_mean, teql_std, teql_len = process_data(teql_data[metric])
        
        if teql_len > 0:
            x = np.arange(0, teql_len * 10, 10)
            
            # Apply smoothing
            teql_mean_smooth = smooth_data(teql_mean, smooth_window)
            teql_std_smooth = smooth_data(teql_std, smooth_window)
            
            config = ALGO_CONFIG['teql']
            
            # Plot mean line
            ax.plot(x, teql_mean_smooth,
                    label=config['label'],
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=2.0)
            
            # Plot confidence band
            ax.fill_between(x,
                           teql_mean_smooth - teql_std_smooth,
                           teql_mean_smooth + teql_std_smooth,
                           alpha=0.2,
                           color=config['color'],
                           edgecolor='none')
            has_data = True
    
    if not has_data:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, color='gray')
    
    # Calculate and display parameter count
    params = calculate_params(env_name, level, mode='tlr-vs-teql')
    
    # Set title with level name and parameter count
    level_name = LEVEL_DISPLAY_NAMES[level]
    ax.set_title(f'{level_name}\n({params:,} params)', fontsize=10, fontweight='bold')
    
    # Configure axes
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Set consistent y-axis limits (0 to 100 with slight extension)
    ax.set_ylim(-5, 105)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_tlr_vs_teql_grid(smooth_window=50, output_base='sensitivity_tlr_vs_teql'):
    """Generate 2x4 grid comparing TLR vs TEQL across all levels."""
    
    set_comparison_style()
    
    print("\n" + "=" * 70)
    print("Plotting TLR vs TEQL Sensitivity Analysis (2x4 Grid)")
    print("=" * 70)
    
    # Check existing results
    check_existing_results('teql')
    check_existing_results('tlr')
    
    # Create figure: 2 rows (Pendulum, CartPole) x 4 columns (levels)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    
    # Environment order (rows)
    envs = ['pendulum', 'cartpole']
    
    for row_idx, env_name in enumerate(envs):
        print(f"\nProcessing {env_name.upper()}...")
        
        for col_idx, level in enumerate(LEVEL_ORDER_COMPARISON):
            ax = axes[row_idx, col_idx]
            plot_single_comparison(ax, env_name, level, smooth_window)
            
            # Add x-label only for bottom row
            if row_idx == 1:
                ax.set_xlabel('Episodes (every 10th)', fontsize=9)
            
            # Add y-label only for first column
            if col_idx == 0:
                ax.set_ylabel(f'{ENV_DISPLAY_NAMES[env_name]}\nCumulative Reward', fontsize=10, fontweight='bold')
            
            # Add legend only for top-left subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=8)
    
    # Adjust layout for compact appearance
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Save plots
    pdf_path = f'figures/{output_base}.pdf'
    png_path = f'figures/{output_base}.png'
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n✓ Saved: {pdf_path}")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ Saved: {png_path}")
    
    return fig


def print_comparison_summary_table():
    """Print summary table of final performance for both algorithms."""
    
    print("\n" + "=" * 100)
    print("TLR vs TEQL SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 100)
    
    for env_name in ['pendulum', 'cartpole']:
        print(f"\n{env_name.upper()}")
        print("-" * 100)
        print(f"{'Level':<15} {'Params':>10} {'TLR Final':>15} {'TLR Std':>10} {'TEQL Final':>15} {'TEQL Std':>10}")
        print("-" * 100)
        
        for level in LEVEL_ORDER_COMPARISON:
            params = calculate_params(env_name, level, mode='tlr-vs-teql')
            
            # Load TLR data
            tlr_data = load_sensitivity_data(env_name, 'tlr', level)
            if tlr_data['greedy_cumulative_reward']:
                tlr_mean, tlr_std, _ = process_data(tlr_data['greedy_cumulative_reward'])
                if len(tlr_mean) > 0:
                    last_n = max(1, len(tlr_mean) // 10)
                    tlr_final = np.mean(tlr_mean[-last_n:])
                    tlr_final_std = np.mean(tlr_std[-last_n:])
                else:
                    tlr_final, tlr_final_std = 0, 0
            else:
                tlr_final, tlr_final_std = 0, 0
            
            # Load TEQL data
            teql_data = load_sensitivity_data(env_name, 'teql', level)
            if teql_data['greedy_cumulative_reward']:
                teql_mean, teql_std, _ = process_data(teql_data['greedy_cumulative_reward'])
                if len(teql_mean) > 0:
                    last_n = max(1, len(teql_mean) // 10)
                    teql_final = np.mean(teql_mean[-last_n:])
                    teql_final_std = np.mean(teql_std[-last_n:])
                else:
                    teql_final, teql_final_std = 0, 0
            else:
                teql_final, teql_final_std = 0, 0
            
            tlr_str = f"{tlr_final:.2f}" if tlr_final != 0 else "N/A"
            tlr_std_str = f"{tlr_final_std:.2f}" if tlr_final_std != 0 else "N/A"
            teql_str = f"{teql_final:.2f}" if teql_final != 0 else "N/A"
            teql_std_str = f"{teql_final_std:.2f}" if teql_final_std != 0 else "N/A"
            
            print(f"{level:<15} {params:>10,} {tlr_str:>15} {tlr_std_str:>10} {teql_str:>15} {teql_std_str:>10}")
        
        print("-" * 100)
    
    print("\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Sensitivity Analysis Plotting (TEQL & TLR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot TEQL-only sensitivity analysis (default)
  python plot_sensitivity.py --mode teql-only
  
  # Plot TLR vs TEQL comparison (2x4 grid)
  python plot_sensitivity.py --mode tlr-vs-teql
  
  # Plot specific environment (TEQL-only mode)
  python plot_sensitivity.py --mode teql-only --env cartpole

  # Plot specific levels (TEQL-only mode)
  python plot_sensitivity.py --mode teql-only --level fine coarse
  
  # Custom smoothing
  python plot_sensitivity.py --mode tlr-vs-teql --smooth 30
  
  # Print summary tables only
  python plot_sensitivity.py --mode teql-only --summary
  python plot_sensitivity.py --mode tlr-vs-teql --summary
Modes:
  teql-only:    Plot TEQL performance across discretization levels (5 levels)
  tlr-vs-teql:  Plot TLR vs TEQL comparison (4 levels, 2x4 grid)
        """
    )
    
    parser.add_argument('--mode', type=str,
                        choices=['teql-only', 'tlr-vs-teql'],
                        default='teql-only',
                        help='Plotting mode: teql-only or tlr-vs-teql')
    parser.add_argument('--env', type=str, nargs='+',
                        choices=['cartpole', 'pendulum', 'all'],
                        default=['all'],
                        help='Environment(s) to plot (TEQL-only mode)')
    parser.add_argument('--level', type=str, nargs='+',
                        choices=['very_fine', 'fine', 'medium', 'coarse', 'very_coarse', 'all'],
                        default=['all'],
                        help='Discretization level(s) to plot (TEQL-only mode)')
    parser.add_argument('--smooth', type=int, default=50,
                        help='Smoothing window size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename base (auto-generated if not specified)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary table only')
    parser.add_argument('--params-only', action='store_true',
                        help='Print parameter counts only (TEQL-only mode)')
    parser.add_argument('--check', action='store_true',
                        help='Only check for existing results')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot')
    
    args = parser.parse_args()
    
    # Handle check-only mode
    if args.check:
        check_existing_results('teql')
        check_existing_results('tlr')
        return
    
    # Handle different modes
    if args.mode == 'teql-only':
        # Determine environments and levels for TEQL-only mode
        if 'all' in args.env:
            envs = ['cartpole', 'pendulum']
        else:
            envs = args.env
        
        if 'all' in args.level:
            levels = LEVEL_ORDER_TEQL
        else:
            levels = args.level
        
        output_base = args.output if args.output else 'sensitivity_analysis'
        
        if args.params_only:
            print_teql_params_only()
        elif args.summary:
            print_teql_summary_table(envs, levels)
        else:
            fig = plot_teql_sensitivity_comparison(envs, levels, args.smooth, output_base)
            print_teql_summary_table(envs, levels)
            
            if not args.no_show:
                plt.show()
    
    elif args.mode == 'tlr-vs-teql':
        output_base = args.output if args.output else 'sensitivity_tlr_vs_teql'
        
        if args.summary:
            print_comparison_summary_table()
        else:
            fig = plot_tlr_vs_teql_grid(args.smooth, output_base)
            print_comparison_summary_table()
            
            if not args.no_show:
                plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()