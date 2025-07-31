import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 设置绘图样式 - 与参考文件完全一致
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,          # 增大基础字体
    'font.family': 'serif',
    'axes.labelsize': 18,     # 增大轴标签字体
    'axes.titlesize': 20,     # 增大标题字体
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'lines.linewidth': 3.5,   # 增大线宽
    'figure.figsize': (16, 10),
    'legend.fontsize': 24,    # 增大图例字体
    'xtick.labelsize': 20,    # 增大x轴刻度字体
    'ytick.labelsize': 20,    # 增大y轴刻度字体
})

def load_json_data(base_dir, env, mode):
    """
    从指定目录加载JSON数据 - 适配iteration_X/结构
    确保正确读取目录结构：results_backup_xxx/iteration_#/{env}_{mode}_tlr_learning.json
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
        # 列出实际存在的目录以帮助调试
        actual_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Actual directories found: {actual_dirs}")
        return data
    
    # 根据iteration数字排序
    iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
    print(f"Found {len(iteration_dirs)} iteration directories: {iteration_dirs}")
    
    # 构建目标文件名 - 根据您的目录结构应该是 {env}_{mode}_tlr_learning.json
    target_filename = f"{env}_{mode}_tlr_learning.json"
    print(f"Looking for files named: {target_filename}")
    
    for iteration_dir in iteration_dirs:
        iteration_path = os.path.join(base_dir, iteration_dir)
        file_path = os.path.join(iteration_path, target_filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist")
            # 列出该iteration目录中实际存在的文件以帮助调试
            if os.path.exists(iteration_path):
                actual_files = [f for f in os.listdir(iteration_path) if f.endswith('.json')]
                print(f"  Actual JSON files in {iteration_path}: {actual_files}")
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
                print(f"  Successfully loaded {len(rewards)} rewards from {file_path}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Final result - loaded {len(data['greedy_steps'])} greedy_steps and {len(data['greedy_cumulative_reward'])} greedy_cumulative_reward")
    
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
    增大窗口大小以获得更平滑的曲线，类似于参考图中的效果
    """
    if len(data) < smooth_window:
        print(f"Warning: Data length {len(data)} is too short for window_size {smooth_window}, using length/2")
        smooth_window = max(1, len(data) // 2)
        
    # 应用两次滤波以获得更平滑的结果，与参考文件一致
    smoothed_once = uniform_filter1d(data, size=smooth_window)
    smoothed_twice = uniform_filter1d(smoothed_once, size=smooth_window//2)
    
    return smoothed_twice

def plot_comparison_single_metric(ax, data_tlr, data_teql, metric, title, env_name, smooth_window=50):
    """
    为单个子图绘制TLR vs TEQL对比（单个metric，单个environment）
    """
    # TLR和TEQL的颜色设置
    colors = {
        'TLR': '#0072B2',    # 蓝色
        'TEQL': '#EC7063'    # 红色
    }
    
    # 处理TLR数据
    tlr_mean, tlr_std, tlr_len = process_data(data_tlr[metric])
    if tlr_len > 0:
        tlr_mean_smooth = apply_post_smoothing(tlr_mean, smooth_window)
        tlr_std_smooth = apply_post_smoothing(tlr_std, smooth_window)
        tlr_x = np.arange(0, tlr_len * 10, 10)
        
        # 绘制TLR曲线和阴影
        ax.plot(tlr_x, tlr_mean_smooth, label='TLR', 
               color=colors['TLR'], linewidth=3.0, linestyle='--')  # TLR用虚线
        ax.fill_between(tlr_x, tlr_mean_smooth - tlr_std_smooth, tlr_mean_smooth + tlr_std_smooth, 
                       alpha=0.25, color=colors['TLR'], edgecolor='none')
    
    # 处理TEQL数据
    teql_mean, teql_std, teql_len = process_data(data_teql[metric])
    if teql_len > 0:
        teql_mean_smooth = apply_post_smoothing(teql_mean, smooth_window)
        teql_std_smooth = apply_post_smoothing(teql_std, smooth_window)
        teql_x = np.arange(0, teql_len * 10, 10)
        
        # 绘制TEQL曲线和阴影
        ax.plot(teql_x, teql_mean_smooth, label='TEQL', 
               color=colors['TEQL'], linewidth=3.0, linestyle='-')  # TEQL用实线
        ax.fill_between(teql_x, teql_mean_smooth - teql_std_smooth, teql_mean_smooth + teql_std_smooth, 
                       alpha=0.25, color=colors['TEQL'], edgecolor='none')
    
    # 设置图形属性
    ax.set_xlabel('Episodes (every 10th)', fontweight='bold', fontsize=18)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=18)
    # 删除子图标题
    # ax.set_title(f'{env_name.capitalize()} - {title}', fontweight='bold', fontsize=20)
    
    # 设置y轴范围为0-100
    ax.set_ylim(0, 110)
    
    # 设置图例
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, facecolor='white', fontsize=16)
    
    # 设置网格和样式
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_tlr_vs_teql_comparison(config_name, config_dir, smooth_window=50):
    """
    为单个离散化标准绘制TLR vs TEQL对比图（2x2布局：两个环境 x 两个metrics）
    """
    environments = ['pendulum', 'cartpole']
    metrics = ['greedy_steps', 'greedy_cumulative_reward']
    metric_titles = ['Greedy Steps', 'Greedy Cumulative Reward']
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 预先加载所有数据
    data_cache = {}
    for env_name in environments:
        data_cache[env_name] = {}
        for algorithm, mode in [('TLR', 'original'), ('TEQL', 'convergent')]:
            print(f"=== Loading {algorithm} data for {env_name} ({config_name}) ===")
            data_cache[env_name][algorithm] = load_json_data(config_dir, env_name, mode)
    
    # 绘制四个子图
    for env_idx, env_name in enumerate(environments):
        for metric_idx, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[env_idx, metric_idx]
            
            tlr_data = data_cache[env_name]['TLR']
            teql_data = data_cache[env_name]['TEQL']
            
            plot_comparison_single_metric(ax, tlr_data, teql_data, metric, metric_title, env_name, smooth_window)
    
    # 添加整体样式调整
    for ax in axes.flat:
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 删除总标题
    # discretization_titles = {
    #     'very_coarse': 'Very Coarse Discretization',
    #     'coarse': 'Coarse Discretization',
    #     'median': 'Median Discretization (Baseline)',
    #     'fine': 'Fine Discretization',
    #     'very_fine': 'Very Fine Discretization'
    # }
    # 
    # fig.suptitle(f'{discretization_titles.get(config_name, config_name.replace("_", " ").title())} - TLR vs TEQL Comparison', 
    #             fontsize=24, fontweight='bold', y=0.98)
    
    # 调整边距
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.25)
    plt.tight_layout()
    
    return fig

def create_all_discretization_comparisons(config_names, config_dirs, smooth_window=50):
    """
    为所有离散化标准创建TLR vs TEQL对比图（每个都是2x2布局）
    """
    # 创建figures文件夹
    os.makedirs('figures', exist_ok=True)
    
    print("="*80)
    print("GENERATING TLR vs TEQL COMPARISON FOR ALL DISCRETIZATION STANDARDS")
    print("="*80)
    
    for config_name, config_dir in zip(config_names, config_dirs):
        print(f"\n{'='*60}")
        print(f"PROCESSING {config_name.upper()} DISCRETIZATION")
        print(f"{'='*60}")
        
        # 生成2x2对比图
        fig = plot_tlr_vs_teql_comparison(config_name, config_dir, smooth_window)
        
        # 保存PDF
        pdf_filename = f'figures/{config_name}_TLR_vs_TEQL_comparison.pdf'
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved: {pdf_filename}")
        
        plt.show()
        plt.close()
    
    print("\n" + "="*80)
    print("ALL DISCRETIZATION COMPARISONS COMPLETE")
    print("="*80)
    print("Generated files:")
    for config_name in config_names:
        print(f"  - figures/{config_name}_TLR_vs_TEQL_comparison.pdf")

def create_summary_grid_comparison(config_names, config_dirs, smooth_window=50):
    """
    创建一个大的网格图，显示所有离散化标准下的TLR vs TEQL对比
    每行代表一个离散化标准，4列分别是：Pendulum Steps, Pendulum Reward, Cartpole Steps, Cartpole Reward
    """
    # TLR和TEQL的颜色设置
    colors = {
        'TLR': '#0072B2',    # 蓝色
        'TEQL': '#EC7063'    # 红色
    }
    
    environments = ['pendulum', 'cartpole']
    metrics = ['greedy_steps', 'greedy_cumulative_reward']
    metric_short_names = ['Steps', 'Reward']
    
    # 创建5行4列的子图 (5个离散化标准 x 4个组合)
    fig, axes = plt.subplots(len(config_names), 4, figsize=(20, 25))
    
    discretization_titles = {
        'very_coarse': 'Very Coarse',
        'coarse': 'Coarse',
        'median': 'Median',  # 更改current为median
        'fine': 'Fine',
        'very_fine': 'Very Fine'
    }
    
    for config_idx, (config_name, config_dir) in enumerate(zip(config_names, config_dirs)):
        print(f"Processing summary for {config_name}...")
        
        # 预加载该配置的所有数据
        data_cache = {}
        for env_name in environments:
            data_cache[env_name] = {}
            for algorithm, mode in [('TLR', 'original'), ('TEQL', 'convergent')]:
                data_cache[env_name][algorithm] = load_json_data(config_dir, env_name, mode)
        
        # 绘制4个子图（2环境 x 2指标）
        col_idx = 0
        for env_idx, env_name in enumerate(environments):
            for metric_idx, (metric, metric_short) in enumerate(zip(metrics, metric_short_names)):
                ax = axes[config_idx, col_idx]
                
                # 处理TLR和TEQL数据
                for algorithm in ['TLR', 'TEQL']:
                    data = data_cache[env_name][algorithm]
                    mean_data, std_data, data_len = process_data(data[metric])
                    
                    if data_len > 0:
                        # 应用平滑
                        mean_smooth = apply_post_smoothing(mean_data, smooth_window)
                        x = np.arange(0, data_len * 10, 10)
                        
                        # 绘制曲线（网格图中不显示阴影以保持清晰）
                        linestyle = '--' if algorithm == 'TLR' else '-'
                        ax.plot(x, mean_smooth, label=algorithm,
                               color=colors[algorithm], linewidth=2.0, linestyle=linestyle)
                
                # 设置y轴范围为0-100
                ax.set_ylim(0, 110)
                
                # 设置图形属性
                if config_idx == len(config_names) - 1:  # 只有底部图显示x轴标签
                    ax.set_xlabel('Episodes', fontsize=12)
                if col_idx == 0:  # 只有最左侧图显示y轴标签
                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                
                # 删除标题（顶部不显示标题）
                # if config_idx == 0:
                #     ax.set_title(f'{env_name.capitalize()} - {metric_short}', fontweight='bold', fontsize=14)
                
                # 只在第一行最后一列显示图例
                if config_idx == 0 and col_idx == 3:
                    ax.legend(loc='lower right', frameon=True, framealpha=0.9,
                             facecolor='white', fontsize=10)
                
                # 设置网格和样式
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(labelsize=9)
                
                col_idx += 1
        
        # 在左侧添加离散化标准标签
        axes[config_idx, 0].text(-0.25, 0.5, discretization_titles.get(config_name, config_name),
                                 rotation=90, verticalalignment='center', fontweight='bold',
                                 fontsize=14, transform=axes[config_idx, 0].transAxes)
    
    # 删除总标题
    # fig.suptitle('TLR vs TEQL Comparison Across All Discretization Standards (Summary Grid)',
    #             fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, left=0.08)
    
    return fig

def main():
    """
    主函数：为每个离散化标准生成TLR vs TEQL对比图（2x2布局）
    """
    # 离散化配置
    config_names = ['very_coarse', 'coarse', 'current', 'fine', 'very_fine']
    config_dirs = ['results_backup_very_coarse', 'results_backup_coarse', 
                   'results_backup_current', 'results_backup_fine', 'results_backup_very_fine']
    
    smooth_window = 50
    
    # 1. 为每个离散化标准生成单独的2x2对比图
    create_all_discretization_comparisons(config_names, config_dirs, smooth_window)
    
    # 2. 生成汇总网格对比图
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY GRID COMPARISON")
    print(f"{'='*60}")
    
    fig = create_summary_grid_comparison(config_names, config_dirs, smooth_window)
    
    # 保存汇总图
    os.makedirs('figures', exist_ok=True)
    summary_filename = 'figures/All_Discretization_TLR_vs_TEQL_Summary.pdf'
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved summary: {summary_filename}")
    
    plt.show()
    plt.close()
    
    print("\n" + "="*80)
    print("ALL COMPARISONS COMPLETE")
    print("="*80)
    print("Generated files:")
    for config_name in config_names:
        print(f"  - figures/{config_name}_TLR_vs_TEQL_comparison.pdf")
    print(f"  - figures/All_Discretization_TLR_vs_TEQL_Summary.pdf")

if __name__ == "__main__":
    main()