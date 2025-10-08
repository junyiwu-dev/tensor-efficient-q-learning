# final_performance_analysis.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (12, 6),
})

def smooth_anomalies(data, window_size=5):
    """Smooth anomalies in the data"""
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

def load_json_data(base_dir, env, algo):
    """Load all iteration data for the specified environment and algorithm"""
    data = {'greedy_cumulative_reward': []}
    
    for iteration_dir in sorted(os.listdir(base_dir), key=lambda x: int(x.split('_')[1]) if 'iteration_' in x else -1):
        if not iteration_dir.startswith('iteration_'):
            continue
            
        file_path = os.path.join(base_dir, iteration_dir, f'{env}_{algo}_tlr_learning.json')
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist, skipping...")
            continue
            
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            
            # Process greedy_cumulative_reward
            rewards = json_data['greedy_cumulative_reward']
            if env == 'pendulum' and len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            rewards = np.array(rewards)
            if len(rewards) == 0:
                print(f"Warning: Empty greedy_cumulative_reward in {file_path}, skipping...")
                continue
                
            # Smoothing
            rewards = smooth_anomalies(rewards)
            data['greedy_cumulative_reward'].append(rewards)
            
    return data

def get_final_performance(reward_data, final_episodes=100):
    """Calculate the average reward of the last several episodes"""
    final_performances = []
    
    for run_rewards in reward_data:
        if len(run_rewards) == 0:
            continue
        
        # Take the average of the last final_episodes episodes
        final_reward = np.mean(run_rewards[-final_episodes:])
        final_performances.append(final_reward)
    
    return final_performances

def plot_final_performance_comparison():
    """Plot final performance comparison boxplot"""
    # Data paths
    original_dir = 'results_backup_TEQL_wo_penalty'  # Original TLR
    penalty_dir = 'results_backup_TEQL_penalty'       # With Penalty
    
    environments = ['pendulum', 'cartpole']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    
    colors = ['#0072B2', '#EC7063', '#009E73']  # Original, With Penalty, Without Penalty
    
    all_statistics = {}
    
    for env_idx, env in enumerate(environments):
        ax = axes[env_idx]
        print(f"\n=== Processing {env.capitalize()} Environment ===")
        
        # Load data
        original_data = load_json_data(original_dir, env, 'original')
        with_penalty_data = load_json_data(penalty_dir, env, 'convergent')
        without_penalty_data = load_json_data(original_dir, env, 'convergent')
        
        # Calculate final performance
        original_final = get_final_performance(original_data['greedy_cumulative_reward'])
        with_penalty_final = get_final_performance(with_penalty_data['greedy_cumulative_reward'])
        without_penalty_final = get_final_performance(without_penalty_data['greedy_cumulative_reward'])
        
        # Prepare boxplot data
        data_to_plot = [original_final, without_penalty_final, with_penalty_final]
        labels = ['Original TLR', 'TEQL w/o Penalty', 'TEQL w/ Penalty']
        
        # Draw boxplot (show mean line)
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                            notch=True, showmeans=True, meanline=True)
        
        # Set colors
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add scatter points
        for i, data in enumerate(data_to_plot):
            if len(data) > 0:
                x = np.random.normal(i+1, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.4, s=20, color=colors[i])
        
        # ax.set_title(f'{env.capitalize()} Environment', fontweight='bold', fontsize=16)
        ax.set_ylabel('Average Reward (Last 100 Episodes)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set special y-axis range for cartpole
        if env == 'cartpole':
            ax.set_ylim(80, None)
        
        # Store statistics
        all_statistics[env] = {
            'Original TLR': {
                'mean': np.mean(original_final) if original_final else 0,
                'std': np.std(original_final) if original_final else 0,
                'median': np.median(original_final) if original_final else 0,
                'n': len(original_final)
            },
            'TEQL w/ Penalty': {
                'mean': np.mean(with_penalty_final) if with_penalty_final else 0,
                'std': np.std(with_penalty_final) if with_penalty_final else 0,
                'median': np.median(with_penalty_final) if with_penalty_final else 0,
                'n': len(with_penalty_final)
            },
            'TEQL w/o Penalty': {
                'mean': np.mean(without_penalty_final) if without_penalty_final else 0,
                'std': np.std(without_penalty_final) if without_penalty_final else 0,
                'median': np.median(without_penalty_final) if without_penalty_final else 0,
                'n': len(without_penalty_final)
            }
        }
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/final_performance_analysis_improved.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    
    # Create statistics summary table
    create_final_performance_tables(all_statistics)
    
    return all_statistics

def create_final_performance_tables(statistics):
    """Create final performance statistics tables"""
    print("\n" + "="*80)
    print("FINAL PERFORMANCE STATISTICS TABLES")
    print("="*80)
    
    # Create table for each environment
    for env, stats in statistics.items():
        print(f"\n{env.upper()} ENVIRONMENT")
        print("-" * 60)
        
        # Create DataFrame
        df_data = []
        for method, values in stats.items():
            df_data.append({
                'Method': method,
                'Mean': f"{values['mean']:.2f}",
                'Std': f"{values['std']:.2f}",
                'Median': f"{values['median']:.2f}",
                'Sample Size': values['n']
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Calculate relative changes
        if stats['Original TLR']['mean'] != 0:
            penalty_change = (stats['TEQL w/ Penalty']['mean'] - stats['Original TLR']['mean']) / stats['Original TLR']['mean'] * 100
            no_penalty_change = (stats['TEQL w/o Penalty']['mean'] - stats['Original TLR']['mean']) / stats['Original TLR']['mean'] * 100
            
            print(f"\nPerformance Changes vs Original TLR:")
            print(f"  TEQL w/ Penalty: {penalty_change:+.1f}%")
            print(f"  TEQL w/o Penalty: {no_penalty_change:+.1f}%")
    
    # Create summary comparison table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print("="*80)
    
    summary_data = []
    for env, stats in statistics.items():
        original_mean = stats['Original TLR']['mean']
        for method, values in stats.items():
            if original_mean != 0 and method != 'Original TLR':
                change = (values['mean'] - original_mean) / original_mean * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "-"
            
            summary_data.append({
                'Environment': env.capitalize(),
                'Method': method,
                'Mean±Std': f"{values['mean']:.1f}±{values['std']:.1f}",
                'Median': f"{values['median']:.1f}",
                'N': values['n'],
                'Change vs Original': change_str
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    plot_final_performance_comparison()