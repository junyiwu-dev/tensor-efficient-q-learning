# comprehensive_analysis_improved.py - Comprehensive analysis file: includes convergence time and final performance analysis
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,          # Increase base font size
    'font.family': 'serif',
    'axes.labelsize': 18,     # Increase axis label font size
    'axes.titlesize': 20,     # Increase title font size
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'legend.fontsize': 16,    # Increase legend font size
    'xtick.labelsize': 14,    # Increase x-axis tick font size
    'ytick.labelsize': 14,    # Increase y-axis tick font size
    'axes.prop_cycle': plt.cycler('color', ['#0072B2', '#009E73','#D2691E'])  # Blue, Red, Green
})

def smooth_anomalies(data, window_size=5):
    """Smooth anomalies in data"""
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
        if abs(data[i] - window_mean) > 0.3 * abs(window_mean):
            smoothed_data[i] = window_mean
    return smoothed_data

def load_json_data(base_dir, env, algo):
    """Read all iteration data for the specified environment and algorithm"""
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

def find_convergence_episodes(reward_data, thresholds=[0.8, 0.9, 0.95]):
    """Find the number of episodes required to reach each performance threshold"""
    convergence_episodes = {threshold: [] for threshold in thresholds}
    
    for run_rewards in reward_data:
        if len(run_rewards) == 0:
            continue
            
        # Calculate final performance (average of last 10%)
        final_performance = np.mean(run_rewards[-max(1, len(run_rewards)//10):])
        
        for threshold in thresholds:
            target_performance = threshold * final_performance
            
            # Find the first episode reaching target performance
            convergence_episode = None
            for i, reward in enumerate(run_rewards):
                if reward >= target_performance:
                    convergence_episode = i * 10  # Convert to actual episode count (recorded every 10 episodes)
                    break
            
            if convergence_episode is not None:
                convergence_episodes[threshold].append(convergence_episode)
            else:
                # If never reached target, record total episode count
                convergence_episodes[threshold].append(len(run_rewards) * 10)
    
    return convergence_episodes

def get_final_performance(reward_data, final_episodes=100):
    """Calculate the average reward of the last few episodes"""
    final_performances = []
    
    for run_rewards in reward_data:
        if len(run_rewards) == 0:
            continue
        
        # Take the average of the last final_episodes episodes
        final_reward = np.mean(run_rewards[-final_episodes:])
        final_performances.append(final_reward)
    
    return final_performances

def plot_convergence_comparison_main():
    """Plot TLR vs TEQL convergence time comparison (main comparison)"""
    # Data paths
    tlr_dir = 'results_backup_TEQL_wo_penalty'  # TLR (original)
    teql_dir = 'results_backup_TEQL_penalty'    # TEQL (convergent with penalty)
    
    environments = ['pendulum', 'cartpole']
    thresholds = [0.8, 0.9, 0.95]
    
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.2, hspace=0.25)
    
    colors = ['#0072B2', '#D2691E']  # TLR blue, TEQL red
    
    all_convergence_stats = {}
    
    for env_idx, env in enumerate(environments):
        print(f"\n=== Processing {env.capitalize()} Environment ===")
        
        # Load data
        tlr_data = load_json_data(tlr_dir, env, 'original')
        teql_data = load_json_data(teql_dir, env, 'convergent')
        
        env_stats = {}
        
        for thresh_idx, threshold in enumerate(thresholds):
            ax = axes[env_idx, thresh_idx]
            
            # Calculate convergence time
            tlr_conv = find_convergence_episodes(tlr_data['greedy_cumulative_reward'], [threshold])[threshold]
            teql_conv = find_convergence_episodes(teql_data['greedy_cumulative_reward'], [threshold])[threshold]
            
            # Prepare boxplot data
            data_to_plot = [tlr_conv, teql_conv]
            labels = ['TLR', 'TEQL']
            
            # Draw boxplot - wider boxes, hide outliers
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                                notch=True, showmeans=True, showfliers=False, widths=0.6)
            
            # Ensure lower bound is not below 0
            for i, whisker in enumerate(box_plot['whiskers']):
                if i % 2 == 0:  # Each box has two whiskers, take the lower whisker
                    ydata = whisker.get_ydata()
                    if ydata[0] < 0:  # If lower bound is below 0
                        whisker.set_ydata([0, 0])  # Set lower bound to 0
            
            # Set colors
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # No title
            ax.set_ylabel('Episodes to Convergence', fontweight='bold', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
            
            # Set x-axis labels slanted, increase font size to avoid overlap
            plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=13)
            
            # Set y-axis range - unified scale
            if env == 'pendulum':
                ax.set_ylim(0, 1000)  # pendulum environment unified to 0-1000
            elif env == 'cartpole':
                ax.set_ylim(0, 2000)  # cartpole environment unified to 0-2000
            
            # Store statistics
            env_stats[threshold] = {
                'TLR': {
                    'data': tlr_conv,
                    'mean': np.mean(tlr_conv) if tlr_conv else 0,
                    'std': np.std(tlr_conv) if tlr_conv else 0,
                    'median': np.median(tlr_conv) if tlr_conv else 0,
                    'n': len(tlr_conv)
                },
                'TEQL': {
                    'data': teql_conv,
                    'mean': np.mean(teql_conv) if teql_conv else 0,
                    'std': np.std(teql_conv) if teql_conv else 0,
                    'median': np.median(teql_conv) if teql_conv else 0,
                    'n': len(teql_conv)
                }
            }
        
        all_convergence_stats[env] = env_stats
    
    plt.tight_layout()
    
    # Create figures folder and save
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/convergence_time_TLR_vs_TEQL.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Convergence comparison plot saved as 'figures/convergence_time_TLR_vs_TEQL.pdf'")
    plt.show()
    
    return all_convergence_stats