import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_json_data(base_dir, env, mode):
    """
    从指定目录加载JSON数据 - 适配iteration_X/结构
    """
    data = {
        'greedy_steps': [],
        'greedy_cumulative_reward': []
    }
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist!")
        return data
    
    print(f"Processing directory: {base_dir}")
    
    # 查找所有iteration目录
    iteration_dirs = [d for d in os.listdir(base_dir) 
                     if d.startswith('iteration_') and os.path.isdir(os.path.join(base_dir, d))]
    
    if not iteration_dirs:
        print(f"No iteration directories found in {base_dir}")
        return data
    
    # 根据iteration数字排序
    iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
    print(f"Found {len(iteration_dirs)} iteration directories")
    
    # 构建目标文件名
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
                
            # 提取 greedy_steps
            if 'greedy_steps' in file_data:
                steps = file_data['greedy_steps']
                if isinstance(steps, list) and len(steps) > 0:
                    data['greedy_steps'].append(steps)
                    
            # 提取 greedy_cumulative_reward
            if 'greedy_cumulative_reward' in file_data:
                rewards = file_data['greedy_cumulative_reward']
                
                # 处理嵌套列表的情况
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
    处理数据，计算均值和标准差，确保长度一致
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
    对平均后的数据应用平滑处理，使用滑动窗口均值滤波
    """
    if len(data) < smooth_window:
        print(f"Warning: Data length {len(data)} is too short for window_size {smooth_window}, using length/2")
        smooth_window = max(1, len(data) // 2)
        
    smoothed = uniform_filter1d(data, size=smooth_window)
    return smoothed

def plot_discretization_sensitivity_combined(algorithm, config_names, config_dirs, 
                                           metric='greedy_cumulative_reward', smooth_window=50):
    """
    绘制discretization sensitivity analysis - 左右两个环境
    """
    # 定义颜色方案 - 从浅到深表示从粗到细的离散化
    colors = ['#FFB6C1', '#FF69B4', '#DC143C', '#B22222', '#8B0000']  # 粉色到深红色渐变
    linestyles = ['-', '-', '-', '-', '-']  # 都用实线
    linewidths = [2.5, 2.5, 3.0, 2.5, 2.5]  # current设置稍粗一些
    
    environments = ['pendulum', 'cartpole']
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for env_idx, env_name in enumerate(environments):
        ax = axes[env_idx]
        
        for i, (config_name, config_dir) in enumerate(zip(config_names, config_dirs)):
            print(f"\n=== Processing {config_name} for {env_name} {metric} ===")
            
            # 根据算法选择mode
            mode = 'original' if algorithm == 'TLR' else 'convergent'
            data = load_json_data(config_dir, env_name, mode)
            
            # 处理数据
            mean_data, std_data, data_len = process_data(data[metric])
            
            if data_len > 0:
                # 应用平滑
                mean_smooth = apply_post_smoothing(mean_data, smooth_window)
                std_smooth = apply_post_smoothing(std_data, smooth_window)
                
                x = np.arange(0, data_len * 10, 10)
                
                # 绘制曲线
                ax.plot(x, mean_smooth, label=config_name.replace('_', ' ').title(), 
                       color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i])
                ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, 
                               alpha=0.2, color=colors[i], edgecolor='none')
            else:
                print(f"Warning: No data found for {config_name}")
        
        # 设置图形属性
        ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=16)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=16)
        ax.set_title(f'{env_name.capitalize()}', fontweight='bold', fontsize=18)
        
        # 设置图例 - 只在第一个子图显示
        if env_idx == 0:
            ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', fontsize=12)
        
        # 设置网格和样式
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """
    主函数：生成两个PDF文件，每个文件包含左右两个环境
    """
    # 离散化配置
    config_names = ['very_coarse', 'coarse', 'current', 'fine', 'very_fine']
    config_dirs = ['results_backup_very_coarse', 'results_backup_coarse', 
                   'results_backup_current', 'results_backup_fine', 'results_backup_very_fine']
    
    smooth_window = 50
    
    # 创建figures文件夹
    os.makedirs('figures', exist_ok=True)
    
    # 1. 生成TLR的discretization sensitivity analysis (Greedy Cumulative Reward)
    print("="*80)
    print("GENERATING TLR DISCRETIZATION SENSITIVITY ANALYSIS")
    print("="*80)
    
    fig = plot_discretization_sensitivity_combined('TLR', config_names, config_dirs, 
                                                  metric='greedy_cumulative_reward', 
                                                  smooth_window=smooth_window)
    
    # 添加总标题
    fig.suptitle('TLR Discretization Sensitivity Analysis - Greedy Cumulative Reward', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # 保存PDF
    pdf_filename = 'figures/TLR_discretization_sensitivity.pdf'
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {pdf_filename}")
    
    plt.show()
    
    # 2. 生成TEQL的discretization sensitivity analysis (Greedy Cumulative Reward)
    print("\n" + "="*80)
    print("GENERATING TEQL DISCRETIZATION SENSITIVITY ANALYSIS")
    print("="*80)
    
    fig = plot_discretization_sensitivity_combined('TEQL', config_names, config_dirs, 
                                                  metric='greedy_cumulative_reward', 
                                                  smooth_window=smooth_window)
    
    # 添加总标题
    fig.suptitle('TEQL Discretization Sensitivity Analysis - Greedy Cumulative Reward', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # 保存PDF
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