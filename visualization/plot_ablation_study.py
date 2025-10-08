import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  

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
    'axes.prop_cycle': plt.cycler('color', ['#D2691E', '#009E73'])  
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
            
            # 处理 greedy_steps
            steps = json_data['greedy_steps']
            # 检查并处理pendulum的greedy_steps数据结构
            if env == 'pendulum' and len(steps) > 0 and isinstance(steps[0], list):
                steps = [s[0] if isinstance(s, list) else s for s in steps]
            
            steps = np.array(steps)
            if len(steps) == 0:
                print(f"Warning: Empty greedy_steps in {file_path}, skipping...")
                continue
                
            # 对每个文件的数据单独进行平滑处理
            steps = smooth_anomalies(steps)
            data['greedy_steps'].append(steps)
            
            # 处理 greedy_cumulative_reward
            rewards = json_data['greedy_cumulative_reward']
            if env == 'pendulum' and len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            rewards = np.array(rewards)
            if len(rewards) == 0:
                print(f"Warning: Empty greedy_cumulative_reward in {file_path}, skipping...")
                continue
                
            # 对每个文件的奖励数据单独进行平滑处理
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
    
    # 应用两次滤波以获得更平滑的结果
    smoothed_once = uniform_filter1d(data, size=smooth_window)
    smoothed_twice = uniform_filter1d(smoothed_once, size=smooth_window//2)
    
    return smoothed_twice

def plot_comparison(ax, data_without_penalty, data_with_penalty, metric, title, env, smooth_window=20):
    # 处理 Without Penalty 数据
    without_mean, without_std, without_len = process_data(data_without_penalty[metric])
    without_x = np.arange(0, without_len * 10, 10) if without_len > 0 else np.array([])
    
    # 处理 With Penalty 数据
    with_mean, with_std, with_len = process_data(data_with_penalty[metric])
    with_x = np.arange(0, with_len * 10, 10) if with_len > 0 else np.array([])

    # 应用后平滑处理
    if without_len > 0:
        without_mean_smooth = apply_post_smoothing(without_mean, smooth_window)
        without_std_smooth = apply_post_smoothing(without_std, smooth_window)
        
    if with_len > 0:
        with_mean_smooth = apply_post_smoothing(with_mean, smooth_window)
        with_std_smooth = apply_post_smoothing(with_std, smooth_window)

    if without_len > 0:
        ax.plot(without_x, without_mean_smooth, label='TEQL Without Penalty', color='#009E73', 
                linewidth=3.0, linestyle='--')  # 虚线样式
        ax.fill_between(without_x, without_mean_smooth - without_std_smooth, without_mean_smooth + without_std_smooth, 
                        alpha=0.25, color='#009E73', edgecolor='none')
        
    if with_len > 0:
        ax.plot(with_x, with_mean_smooth, label='TEQL With Penalty', color='#D2691E', 
                linewidth=3.0, linestyle='-')  # 实线样式
        ax.fill_between(with_x, with_mean_smooth - with_std_smooth, with_mean_smooth + with_std_smooth, 
                        alpha=0.25, color='#D2691E', edgecolor='none')

    ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=24)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=24)
    
    # 设置图例位置和样式 - 统一固定在右下角
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', fontsize=20)
    
    # 设置网格线风格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 增大刻度字体
    ax.tick_params(labelsize=14)

def main():
    base_dir = 'results_backup_TEQL_wo_penalty'  # First results folder
    penalty_dir = 'results_backup_TEQL_penalty'  # Second results folder (with penalty)

    smooth_window = 50  # Smoothing window size

    # Load Without Penalty data
    print("\n=== Processing Pendulum Without Penalty (results_backup) ===")
    pendulum_without_penalty = load_json_data(base_dir, 'pendulum', 'convergent')
    
    print("\n=== Processing Cartpole Without Penalty (results_backup) ===")
    cartpole_without_penalty = load_json_data(base_dir, 'cartpole', 'convergent')
    
    # Load With Penalty data
    print("\n=== Processing Pendulum With Penalty (results_backup) ===")
    pendulum_with_penalty = load_json_data(penalty_dir, 'pendulum', 'convergent')
    
    print("\n=== Processing Cartpole With Penalty (results_backup) ===")
    cartpole_with_penalty = load_json_data(penalty_dir, 'cartpole', 'convergent')

    # Create plots: 2 rows x 2 columns, add overall title
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Increase figure size
    # fig.suptitle('Comparison of TEQL Learning Approaches', fontsize=24, fontweight='bold', y=0.98)
    
    # Adjust margins
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.25)

    # Pendulum - Greedy Steps
    plot_comparison(axes[0, 0], pendulum_without_penalty, pendulum_with_penalty, 'greedy_steps', 'Greedy Steps', 'Pendulum', smooth_window)

    # Pendulum - Greedy Cumulative Reward
    plot_comparison(axes[0, 1], pendulum_without_penalty, pendulum_with_penalty, 'greedy_cumulative_reward', 'Greedy Cumulative Reward', 'Pendulum', smooth_window)

    # Cartpole - Greedy Steps
    plot_comparison(axes[1, 0], cartpole_without_penalty, cartpole_with_penalty, 'greedy_steps', 'Greedy Steps', 'Cartpole', smooth_window)

    # Cartpole - Greedy Cumulative Reward
    plot_comparison(axes[1, 1], cartpole_without_penalty, cartpole_with_penalty, 'greedy_cumulative_reward', 'Greedy Cumulative Reward', 'Cartpole', smooth_window)

    # Better axis labels
    for ax in axes.flat:
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()  # Reserve top space for overall title
    
    # Create figures folder
    os.makedirs('figures', exist_ok=True)
    
    # Save as high-res PDF
    plt.savefig('figures/comparison_TEQL_with_vs_without_penalty.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Plot saved as 'figures/comparison_TEQL_with_vs_without_penalty.pdf'")
    
    plt.show()

if __name__ == "__main__":
    main()