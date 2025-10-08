import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # For smoothing

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
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
    'legend.fontsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.prop_cycle': plt.cycler('color', ['#0072B2', '#EC7063'])  # Blue and red
})

def smooth_anomalies(data, window_size=5):
    if len(data) < 2 * window_size + 1:
        print(f"Warning: Data length {len(data)} is too short for window_size {window_size}, skipping smoothing")
        return data.copy()
        
    smoothed_data = data.copy()
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        
        window_values = data[start:i].tolist() + data[i+1:end].tolist()
        if not window_values:
            continue
        
        window_mean = np.mean(window_values)
        
        # If the current point differs too much from the window mean (threshold reduced from 50% to 30%)
        if abs(data[i] - window_mean) > 0.3 * abs(window_mean):
            smoothed_data[i] = window_mean
    
    return smoothed_data

def load_json_data(base_dir, env, algo):
    data = {'greedy_steps': [], 'greedy_cumulative_reward': []}
    
    for iteration_dir in sorted(os.listdir(base_dir), key=lambda x: int(x.split('_')[1]) if 'iteration_' in x else -1):
        if not iteration_dir.startswith('iteration_'):
            continue
            
        file_path = os.path.join(base_dir, iteration_dir, f'{env}_{algo}_tlr_learning.json')
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist, skipping...")
            continue
            
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            
            # Process greedy_steps
            steps = json_data['greedy_steps']
            # Check and process pendulum's greedy_steps data structure
            if env == 'pendulum' and len(steps) > 0 and isinstance(steps[0], list):
                steps = [s[0] if isinstance(s, list) else s for s in steps]
            
            steps = np.array(steps)
            if len(steps) == 0:
                print(f"Warning: Empty greedy_steps in {file_path}, skipping...")
                continue
                
            # Smooth each file's data individually
            steps = smooth_anomalies(steps)
            data['greedy_steps'].append(steps)
            
            # Process greedy_cumulative_reward
            rewards = json_data['greedy_cumulative_reward']
            if env == 'pendulum' and len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            rewards = np.array(rewards)
            if len(rewards) == 0:
                print(f"Warning: Empty greedy_cumulative_reward in {file_path}, skipping...")
                continue
                
            # Smooth each file's reward data individually
            rewards = smooth_anomalies(rewards)
            data['greedy_cumulative_reward'].append(rewards)
            
    return data

def process_data(data_list):
    if not data_list:
        print("Warning: No data to process")
        return np.array([]), np.array([]), 0
        
    min_len = min(len(data) for data in data_list)
    trimmed_data = [data[:min_len] for data in data_list]
    data_array = np.array(trimmed_data)
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    
    return mean_data, std_data, min_len

def apply_post_smoothing(data, smooth_window=20):
    if len(data) < smooth_window:
        print(f"Warning: Data length {len(data)} is too short for window_size {smooth_window}, skipping smoothing")
        return data
        
    # Apply filter twice for smoother results
    smoothed_once = uniform_filter1d(data, size=smooth_window)
    smoothed_twice = uniform_filter1d(smoothed_once, size=smooth_window//2)
    
    return smoothed_twice

def plot_comparison(ax, data_original, data_with_penalty, metric, title, env, smooth_window=20):
    # Process Original data
    orig_mean, orig_std, orig_len = process_data(data_original[metric])
    orig_x = np.arange(0, orig_len * 10, 10) if orig_len > 0 else np.array([])

    # Process With Penalty data
    with_mean, with_std, with_len = process_data(data_with_penalty[metric])
    with_x = np.arange(0, with_len * 10, 10) if with_len > 0 else np.array([])

    # Apply post-smoothing with larger window
    if orig_len > 0:
        orig_mean_smooth = apply_post_smoothing(orig_mean, smooth_window)
        orig_std_smooth = apply_post_smoothing(orig_std, smooth_window)
    
    if with_len > 0:
        with_mean_smooth = apply_post_smoothing(with_mean, smooth_window)
        with_std_smooth = apply_post_smoothing(with_std, smooth_window)

    # Plot Original TLR - blue dashed line (colorblind friendly)
    if orig_len > 0:
        ax.plot(orig_x, orig_mean_smooth, label='TLR', color='#0072B2', 
                linewidth=3.0, linestyle='--')
        ax.fill_between(orig_x, orig_mean_smooth - orig_std_smooth, orig_mean_smooth + orig_std_smooth, 
                        alpha=0.25, color='#0072B2', edgecolor='none')

    # Plot With Penalty TLR - red solid line
    if with_len > 0:
        ax.plot(with_x, with_mean_smooth, label='TEQL', color='#D2691E', 
                linewidth=3.0, linestyle='-')
        ax.fill_between(with_x, with_mean_smooth - with_std_smooth, with_mean_smooth + with_std_smooth, 
                        alpha=0.25, color='#D2691E', edgecolor='none')

    ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=24)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=24)
    # ax.set_title(f'{env} - {title}', fontweight='bold', fontsize=20)
    
    # Set legend position and style - always at lower right
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', fontsize=20)
    
    # Set grid style
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Increase tick font size
    ax.tick_params(labelsize=14)

def main():
    base_dir = 'results_backup_very_fine'  # First results folder
    penalty_dir = 'results_backup_very_fine'  # Second results folder (with penalty)

    smooth_window = 50  # Increase smoothing window for smoother curves

    # Load results_backup data and apply smoothing
    print("\n=== Processing Pendulum Original (results_backup) ===")
    pendulum_original = load_json_data(base_dir, 'pendulum', 'original')
    
    print("\n=== Processing Cartpole Original (results_backup) ===")
    cartpole_original = load_json_data(base_dir, 'cartpole', 'original')
    
    # Load results_backup_UCB_maxiter5_penalty data and apply smoothing
    print("\n=== Processing Pendulum With Penalty (results_backup_UCB_maxiter5_penalty) ===")
    pendulum_with_penalty = load_json_data(penalty_dir, 'pendulum', 'convergent')
    
    print("\n=== Processing Cartpole With Penalty (results_backup_UCB_maxiter5_penalty) ===")
    cartpole_with_penalty = load_json_data(penalty_dir, 'cartpole', 'convergent')

    # Create plot: 2 rows x 2 columns, add overall title
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    # fig.suptitle('Comparison of TLR Learning Approaches', fontsize=24, fontweight='bold', y=0.98)
    
    # Add margins for better appearance
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.25)

    # Pendulum - Greedy Steps
    plot_comparison(axes[0, 0], pendulum_original, pendulum_with_penalty, 'greedy_steps', 'Greedy Steps', 'Pendulum', smooth_window)

    # Pendulum - Greedy Cumulative Reward
    plot_comparison(axes[0, 1], pendulum_original, pendulum_with_penalty, 'greedy_cumulative_reward', 'Greedy Cumulative Reward', 'Pendulum', smooth_window)

    # Cartpole - Greedy Steps
    plot_comparison(axes[1, 0], cartpole_original, cartpole_with_penalty, 'greedy_steps', 'Greedy Steps', 'Cartpole', smooth_window)

    # Cartpole - Greedy Cumulative Reward
    plot_comparison(axes[1, 1], cartpole_original, cartpole_with_penalty, 'greedy_cumulative_reward', 'Greedy Cumulative Reward', 'Cartpole', smooth_window)

    # Better axis labels
    for ax in axes.flat:
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Create figures folder
    os.makedirs('figures', exist_ok=True)
    
    # Save as high-res PDF
    plt.savefig('figures/comparison_TLR_vs_TEQL.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Plot saved as 'figures/comparison_TLR_vs_TEQL.pdf'")
    
    plt.show()

if __name__ == "__main__":
    main()