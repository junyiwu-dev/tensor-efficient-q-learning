import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # 用于平滑处理

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 22,          # 增大基础字体
    'font.family': 'serif',
    'axes.labelsize': 26,     # 增大轴标签字体
    'axes.titlesize': 30,     # 增大标题字体
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'lines.linewidth': 3.0,   # 增大线宽
    'figure.figsize': (16, 10),
    'legend.fontsize': 22,    # 增大图例字体
    'xtick.labelsize': 20,    # 增大x轴刻度字体
    'ytick.labelsize': 20,    # 增大y轴刻度字体
    'axes.prop_cycle': plt.cycler('color', ['#0072B2', '#EC7063'])  # 蓝色和红色
})

def smooth_anomalies(data, window_size=5):
    """
    使用异常值平滑方法，如果一个点与周围点相差太大，使用周围点的均值替代
    window_size 定义了单侧窗口大小，总窗口为 2*window_size+1
    增大默认窗口大小以提高平滑效果
    """
    if len(data) < 2 * window_size + 1:  # 数据长度不足以平滑
        print(f"Warning: Data length {len(data)} is too short for window_size {window_size}, skipping smoothing")
        return data.copy()
        
    smoothed_data = data.copy()
    for i in range(len(data)):
        # 获取窗口范围
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        
        # 计算窗口内除当前点外的均值
        window_values = data[start:i].tolist() + data[i+1:end].tolist()  # 使用 list 拼接避免形状问题
        if not window_values:  # 如果窗口为空，继续
            continue
        
        window_mean = np.mean(window_values)
        
        # 如果当前点与窗口均值相差太大（减小阈值从50%到30%以捕捉更多异常值）
        if abs(data[i] - window_mean) > 0.3 * abs(window_mean):
            smoothed_data[i] = window_mean
    
    return smoothed_data

def load_json_data(base_dir, env, algo):
    """
    读取指定环境和算法的所有 iteration 数据
    """
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

def apply_post_smoothing(data, smooth_window=20):
    """
    对平均后的数据应用平滑处理，使用滑动窗口均值滤波
    增大窗口大小以获得更平滑的曲线，类似于参考图中的效果
    """
    if len(data) < smooth_window:
        print(f"Warning: Data length {len(data)} is too short for window_size {smooth_window}, skipping smoothing")
        return data
        
    # 应用两次滤波以获得更平滑的结果
    smoothed_once = uniform_filter1d(data, size=smooth_window)
    smoothed_twice = uniform_filter1d(smoothed_once, size=smooth_window//2)
    
    return smoothed_twice

def plot_comparison(ax, data_original, data_with_penalty, metric, title, env, smooth_window=20):
    """
    绘制比较图，包括均值和标准差阴影，并应用均值后平滑处理
    仅包含两条曲线：original, with penalty
    """
    # 处理 Original 数据
    orig_mean, orig_std, orig_len = process_data(data_original[metric])
    orig_x = np.arange(0, orig_len * 10, 10) if orig_len > 0 else np.array([])

    # 处理 With Penalty 数据
    with_mean, with_std, with_len = process_data(data_with_penalty[metric])
    with_x = np.arange(0, with_len * 10, 10) if with_len > 0 else np.array([])

    # 应用后平滑处理，使用更大的窗口
    if orig_len > 0:
        orig_mean_smooth = apply_post_smoothing(orig_mean, smooth_window)
        orig_std_smooth = apply_post_smoothing(orig_std, smooth_window)
    
    if with_len > 0:
        with_mean_smooth = apply_post_smoothing(with_mean, smooth_window)
        with_std_smooth = apply_post_smoothing(with_std, smooth_window)

    # 绘制 Original TLR - 使用蓝色虚线（为色盲友好）
    if orig_len > 0:
        ax.plot(orig_x, orig_mean_smooth, label='TLR', color='#0072B2', 
                linewidth=3.0, linestyle='--')  # 虚线样式
        ax.fill_between(orig_x, orig_mean_smooth - orig_std_smooth, orig_mean_smooth + orig_std_smooth, 
                        alpha=0.25, color='#0072B2', edgecolor='none')

    # 绘制 With Penalty TLR - 使用红色实线
    if with_len > 0:
        ax.plot(with_x, with_mean_smooth, label='TEQL', color='#D2691E', 
                linewidth=3.0, linestyle='-')  # 实线样式
        ax.fill_between(with_x, with_mean_smooth - with_std_smooth, with_mean_smooth + with_std_smooth, 
                        alpha=0.25, color='#D2691E', edgecolor='none')

    ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=24)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=24)
    # ax.set_title(f'{env} - {title}', fontweight='bold', fontsize=20)
    
    # 设置y轴范围为0-100
    ax.set_ylim(0, 100)
    
    # 设置图例位置和样式 - 统一固定在右下角
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', fontsize=20)
    
    # 设置网格线风格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 增大刻度字体
    ax.tick_params(labelsize=14)

def main():
    base_dir = 'results_backup_current'  # 第一个结果文件夹
    penalty_dir = 'results_backup_current'  # 第二个结果文件夹（带惩罚项）

    smooth_window = 50  # 增大平滑窗口大小以获得更平滑的曲线

    # 读取 results_backup 数据并应用平滑处理
    print("\n=== Processing Pendulum Original (results_backup) ===")
    pendulum_original = load_json_data(base_dir, 'pendulum', 'original')
    
    print("\n=== Processing Cartpole Original (results_backup) ===")
    cartpole_original = load_json_data(base_dir, 'cartpole', 'original')
    
    # 读取 results_backup_UCB_maxiter5_penalty 数据并应用平滑处理
    print("\n=== Processing Pendulum With Penalty (results_backup_UCB_maxiter5_penalty) ===")
    pendulum_with_penalty = load_json_data(penalty_dir, 'pendulum', 'convergent')
    
    print("\n=== Processing Cartpole With Penalty (results_backup_UCB_maxiter5_penalty) ===")
    cartpole_with_penalty = load_json_data(penalty_dir, 'cartpole', 'convergent')

    # 创建绘图：1 行 2 列，左为pendulum，右为cartpole
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # 调整为1行2列
    # fig.suptitle('Comparison of TLR Learning Approaches', fontsize=24, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.25)

    # Pendulum - Greedy Cumulative Reward（左）
    plot_comparison(axes[0], pendulum_original, pendulum_with_penalty, 'greedy_cumulative_reward', 'Greedy Cumulative Reward', 'Pendulum', smooth_window)

    # Cartpole - Greedy Cumulative Reward（右）
    plot_comparison(axes[1], cartpole_original, cartpole_with_penalty, 'greedy_cumulative_reward', 'Greedy Cumulative Reward', 'Cartpole', smooth_window)

    # 添加更好的坐标轴标签
    for ax in axes.flat:
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()  # 保留顶部空间给整体标题
    
    # 创建figures文件夹
    os.makedirs('figures', exist_ok=True)
    
    # 保存为高清PDF
    plt.savefig('figures/comparison_TLR_vs_TEQL.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Plot saved as 'figures/comparison_TLR_vs_TEQL.pdf'")
    
    plt.show()

if __name__ == "__main__":
    main()