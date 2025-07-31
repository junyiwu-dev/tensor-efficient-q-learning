# comprehensive_analysis_improved.py - 综合分析文件：包含收敛时间和最终性能分析
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,          # 增大基础字体
    'font.family': 'serif',
    'axes.labelsize': 18,     # 增大轴标签字体
    'axes.titlesize': 20,     # 增大标题字体
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'legend.fontsize': 16,    # 增大图例字体
    'xtick.labelsize': 14,    # 增大x轴刻度字体
    'ytick.labelsize': 14,    # 增大y轴刻度字体
    'axes.prop_cycle': plt.cycler('color', ['#0072B2', '#009E73','#D2691E'])  # 蓝色、红色、绿色
})

def smooth_anomalies(data, window_size=5):
    """平滑异常值"""
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
    """读取指定环境和算法的所有 iteration 数据"""
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
            
            # 处理 greedy_cumulative_reward
            rewards = json_data['greedy_cumulative_reward']
            if env == 'pendulum' and len(rewards) > 0 and isinstance(rewards[0], list):
                rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            
            rewards = np.array(rewards)
            if len(rewards) == 0:
                print(f"Warning: Empty greedy_cumulative_reward in {file_path}, skipping...")
                continue
                
            # 平滑处理
            rewards = smooth_anomalies(rewards)
            data['greedy_cumulative_reward'].append(rewards)
            
    return data

def find_convergence_episodes(reward_data, thresholds=[0.8, 0.9, 0.95]):
    """找到达到各个性能阈值所需的回合数"""
    convergence_episodes = {threshold: [] for threshold in thresholds}
    
    for run_rewards in reward_data:
        if len(run_rewards) == 0:
            continue
            
        # 计算最终性能（最后10%的平均值）
        final_performance = np.mean(run_rewards[-max(1, len(run_rewards)//10):])
        
        for threshold in thresholds:
            target_performance = threshold * final_performance
            
            # 找到首次达到目标性能的回合
            convergence_episode = None
            for i, reward in enumerate(run_rewards):
                if reward >= target_performance:
                    convergence_episode = i * 10  # 转换为实际回合数（每10个回合记录一次）
                    break
            
            if convergence_episode is not None:
                convergence_episodes[threshold].append(convergence_episode)
            else:
                # 如果从未达到目标，记录总回合数
                convergence_episodes[threshold].append(len(run_rewards) * 10)
    
    return convergence_episodes

def get_final_performance(reward_data, final_episodes=100):
    """计算最后几个回合的平均奖励"""
    final_performances = []
    
    for run_rewards in reward_data:
        if len(run_rewards) == 0:
            continue
        
        # 取最后final_episodes个回合的平均值
        final_reward = np.mean(run_rewards[-final_episodes:])
        final_performances.append(final_reward)
    
    return final_performances

def plot_convergence_comparison_main():
    """绘制TLR vs TEQL收敛时间对比图（主要对比）"""
    # 数据路径
    tlr_dir = 'results_backup_TEQL_wo_penalty'  # TLR (original)
    teql_dir = 'results_backup_TEQL_penalty'    # TEQL (convergent with penalty)
    
    environments = ['pendulum', 'cartpole']
    thresholds = [0.8, 0.9, 0.95]
    
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.2, hspace=0.25)
    
    colors = ['#0072B2', '#D2691E']  # TLR蓝色, TEQL红色
    
    all_convergence_stats = {}
    
    for env_idx, env in enumerate(environments):
        print(f"\n=== Processing {env.capitalize()} Environment ===")
        
        # 加载数据
        tlr_data = load_json_data(tlr_dir, env, 'original')
        teql_data = load_json_data(teql_dir, env, 'convergent')
        
        env_stats = {}
        
        for thresh_idx, threshold in enumerate(thresholds):
            ax = axes[env_idx, thresh_idx]
            
            # 计算收敛时间
            tlr_conv = find_convergence_episodes(tlr_data['greedy_cumulative_reward'], [threshold])[threshold]
            teql_conv = find_convergence_episodes(teql_data['greedy_cumulative_reward'], [threshold])[threshold]
            
            # 准备箱线图数据
            data_to_plot = [tlr_conv, teql_conv]
            labels = ['TLR', 'TEQL']
            
            # 绘制箱线图 - 更宽的箱子，隐藏outliers
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                                notch=True, showmeans=True, showfliers=False, widths=0.6)
            
            # 处理lower bound不低于0的问题
            for i, whisker in enumerate(box_plot['whiskers']):
                if i % 2 == 0:  # 每个箱子有两个whiskers，取下方的whisker
                    ydata = whisker.get_ydata()
                    if ydata[0] < 0:  # 如果下边界低于0
                        whisker.set_ydata([0, 0])  # 将下边界设为0
            
            # 设置颜色
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 不设置标题
            ax.set_ylabel('Episodes to Convergence', fontweight='bold', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
            
            # 设置x轴标签为倾斜排列，增大字体以避免重叠
            plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=13)
            
            # 设置y轴范围 - 统一scale
            if env == 'pendulum':
                ax.set_ylim(0, 1000)  # pendulum环境统一为0-1000
            elif env == 'cartpole':
                ax.set_ylim(0, 2000)  # cartpole环境统一为0-2000
            
            # 存储统计信息
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
    
    # 创建figures文件夹并保存
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/convergence_time_TLR_vs_TEQL.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Convergence comparison plot saved as 'figures/convergence_time_TLR_vs_TEQL.pdf'")
    plt.show()
    
    return all_convergence_stats

def plot_convergence_comparison_ablation():
    """绘制TEQL收敛时间对比图（消融研究）"""
    # 数据路径
    tlr_dir = 'results_backup_TEQL_wo_penalty'     # TLR (original)
    teql_penalty_dir = 'results_backup_TEQL_penalty'  # TEQL with penalty
    
    environments = ['pendulum', 'cartpole']
    thresholds = [0.8, 0.9, 0.95]
    
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.15, wspace=0.2, hspace=0.25)
    
    colors = ['#0072B2','#009E73','#D2691E']  # TLR, TEQL w/o penalty, TEQL w/ penalty
    
    all_convergence_stats = {}
    
    for env_idx, env in enumerate(environments):
        print(f"\n=== Processing {env.capitalize()} Environment ===")
        
        # 加载数据
        tlr_data = load_json_data(tlr_dir, env, 'original')
        teql_with_penalty_data = load_json_data(teql_penalty_dir, env, 'convergent')
        teql_without_penalty_data = load_json_data(tlr_dir, env, 'convergent')
        
        env_stats = {}
        
        for thresh_idx, threshold in enumerate(thresholds):
            ax = axes[env_idx, thresh_idx]
            
            # 计算收敛时间
            tlr_conv = find_convergence_episodes(tlr_data['greedy_cumulative_reward'], [threshold])[threshold]
            teql_without_penalty_conv = find_convergence_episodes(teql_without_penalty_data['greedy_cumulative_reward'], [threshold])[threshold]
            teql_with_penalty_conv = find_convergence_episodes(teql_with_penalty_data['greedy_cumulative_reward'], [threshold])[threshold]
            
            # 准备箱线图数据 - 从高到低排列
            data_to_plot = [tlr_conv, teql_without_penalty_conv, teql_with_penalty_conv]
            labels = ['TLR', 'TEQL w/o Penalty', 'TEQL w/ Penalty']
            
            # 绘制箱线图 - 更宽的箱子，隐藏outliers
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                                notch=True, showmeans=True, showfliers=False, widths=0.6)
            
            # 处理lower bound不低于0的问题
            for i, whisker in enumerate(box_plot['whiskers']):
                if i % 2 == 0:  # 每个箱子有两个whiskers，取下方的whisker
                    ydata = whisker.get_ydata()
                    if ydata[0] < 0:  # 如果下边界低于0
                        whisker.set_ydata([0, 0])  # 将下边界设为0
            
            # 设置颜色
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 不设置标题
            ax.set_ylabel('Episodes to Convergence', fontweight='bold', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
            
            # 设置x轴标签为倾斜排列，增大字体以避免重叠
            plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=13)
            
            # 设置y轴范围 - 统一scale
            if env == 'pendulum':
                ax.set_ylim(0, 1000)  # pendulum环境统一为0-1000
            elif env == 'cartpole':
                ax.set_ylim(0, 2000)  # cartpole环境统一为0-2000
        
        all_convergence_stats[env] = env_stats
    
    plt.tight_layout()
    
    # 创建figures文件夹并保存
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/convergence_time_analysis_ablation.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Convergence ablation plot saved as 'figures/convergence_time_analysis_ablation.pdf'")
    plt.show()
    
    return all_convergence_stats

def plot_final_performance_main():
    """绘制TLR vs TEQL最终性能对比箱线图（主要对比）"""
    # 数据路径
    tlr_dir = 'results_backup_TEQL_wo_penalty'  # TLR (Original)
    teql_dir = 'results_backup_TEQL_penalty'    # TEQL (Convergent with penalty)
    
    environments = ['pendulum', 'cartpole']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.12, wspace=0.2)
    
    colors = ['#0072B2', '#D2691E']  # TLR蓝色, TEQL红色
    
    all_statistics = {}
    
    for env_idx, env in enumerate(environments):
        ax = axes[env_idx]
        print(f"\n=== Processing {env.capitalize()} Environment ===")
        
        # 加载数据
        tlr_data = load_json_data(tlr_dir, env, 'original')
        teql_data = load_json_data(teql_dir, env, 'convergent')
        
        # 计算最终性能
        tlr_final = get_final_performance(tlr_data['greedy_cumulative_reward'])
        teql_final = get_final_performance(teql_data['greedy_cumulative_reward'])
        
        # 准备箱线图数据
        data_to_plot = [tlr_final, teql_final]
        labels = ['TLR', 'TEQL']
        
        # 绘制箱线图 - 更宽的箱子，隐藏outliers
        box_plot = ax.boxplot(
            data_to_plot, labels=labels, patch_artist=True,
            notch=True, showmeans=True, meanline=True, showfliers=False, widths=0.6
        )
        
        # 处理lower bound不低于0的问题
        for i, whisker in enumerate(box_plot['whiskers']):
            if i % 2 == 0:  # 每个箱子有两个whiskers，取下方的whisker
                ydata = whisker.get_ydata()
                if ydata[0] < 0:  # 如果下边界低于0
                    whisker.set_ydata([0, 0])  # 将下边界设为0
        
        # 设置颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 不设置标题
        ax.set_ylabel('Average Reward (Last 100 Episodes)', fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        
        # 设置x轴标签为水平排列，保持当前字体大小
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # 设置y轴范围 - 更紧凑的显示
        if env == 'cartpole':
            # 对cartpole设置特殊范围87-90
            ax.set_ylim(87, 90)
        else:
            # 对pendulum设置紧凑范围
            if len(tlr_final) > 0 and len(teql_final) > 0:
                min_val = min(min(tlr_final), min(teql_final))
                max_val = max(max(tlr_final), max(teql_final))
                margin = (max_val - min_val) * 0.05
                ax.set_ylim(min_val - margin, max_val + margin)
        
        # 存储统计信息
        all_statistics[env] = {
            'TLR': {
                'mean': np.mean(tlr_final) if tlr_final else 0,
                'std': np.std(tlr_final) if tlr_final else 0,
                'median': np.median(tlr_final) if tlr_final else 0,
                'n': len(tlr_final)
            },
            'TEQL': {
                'mean': np.mean(teql_final) if teql_final else 0,
                'std': np.std(teql_final) if teql_final else 0,
                'median': np.median(teql_final) if teql_final else 0,
                'n': len(teql_final)
            }
        }
    
    plt.tight_layout()
    
    # 创建figures文件夹并保存
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/final_performance_TLR_vs_TEQL.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Final performance comparison plot saved as 'figures/final_performance_TLR_vs_TEQL.pdf'")
    plt.show()
    
    return all_statistics

def plot_final_performance_ablation():
    """绘制TEQL最终性能对比箱线图（消融研究）"""
    # 数据路径
    tlr_dir = 'results_backup_TEQL_wo_penalty'     # TLR (original)
    teql_penalty_dir = 'results_backup_TEQL_penalty'  # TEQL with penalty
    
    environments = ['pendulum', 'cartpole']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.18, wspace=0.2)
    
    colors = ['#0072B2','#009E73','#D2691E']  # TLR, TEQL w/o penalty, TEQL w/ penalty
    
    all_statistics = {}
    
    for env_idx, env in enumerate(environments):
        ax = axes[env_idx]
        print(f"\n=== Processing {env.capitalize()} Environment ===")
        
        # 加载数据
        tlr_data = load_json_data(tlr_dir, env, 'original')
        teql_with_penalty_data = load_json_data(teql_penalty_dir, env, 'convergent')
        teql_without_penalty_data = load_json_data(tlr_dir, env, 'convergent')
        
        # 计算最终性能
        tlr_final = get_final_performance(tlr_data['greedy_cumulative_reward'])
        teql_without_penalty_final = get_final_performance(teql_without_penalty_data['greedy_cumulative_reward'])
        teql_with_penalty_final = get_final_performance(teql_with_penalty_data['greedy_cumulative_reward'])
        
        # 准备箱线图数据 - 从高到低排列
        data_to_plot = [tlr_final, teql_without_penalty_final, teql_with_penalty_final]
        labels = ['TLR', 'TEQL w/o Penalty', 'TEQL w/ Penalty']
        
        # 绘制箱线图 - 更宽的箱子，隐藏outliers  
        box_plot = ax.boxplot(
            data_to_plot, labels=labels, patch_artist=True,
            notch=True, showmeans=True, meanline=True, showfliers=False, widths=0.6
        )
        
        # 处理lower bound不低于0的问题
        for i, whisker in enumerate(box_plot['whiskers']):
            if i % 2 == 0:  # 每个箱子有两个whiskers，取下方的whisker
                ydata = whisker.get_ydata()
                if ydata[0] < 0:  # 如果下边界低于0
                    whisker.set_ydata([0, 0])  # 将下边界设为0
        
        # 设置颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 不设置标题
        ax.set_ylabel('Average Reward (Last 100 Episodes)', fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        
        # 设置x轴标签为倾斜排列，缩小字体以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
        
        # 设置y轴范围 - 更紧凑的显示  
        if env == 'cartpole':
            # 对cartpole设置特殊范围87-90
            ax.set_ylim(87, 90)
        else:
            # 对pendulum设置紧凑范围
            if len(tlr_final) > 0 and len(teql_with_penalty_final) > 0:
                min_val = min(min(tlr_final), min(teql_without_penalty_final), min(teql_with_penalty_final))
                max_val = max(max(tlr_final), max(teql_without_penalty_final), max(teql_with_penalty_final))
                margin = (max_val - min_val) * 0.05
                ax.set_ylim(min_val - margin, max_val + margin)
        
        # 存储统计信息
        all_statistics[env] = {
            'TLR': {
                'mean': np.mean(tlr_final) if tlr_final else 0,
                'std': np.std(tlr_final) if tlr_final else 0,
                'median': np.median(tlr_final) if tlr_final else 0,
                'n': len(tlr_final)
            },
            'TEQL w/o Penalty': {
                'mean': np.mean(teql_without_penalty_final) if teql_without_penalty_final else 0,
                'std': np.std(teql_without_penalty_final) if teql_without_penalty_final else 0,
                'median': np.median(teql_without_penalty_final) if teql_without_penalty_final else 0,
                'n': len(teql_without_penalty_final)
            },
            'TEQL w/ Penalty': {
                'mean': np.mean(teql_with_penalty_final) if teql_with_penalty_final else 0,
                'std': np.std(teql_with_penalty_final) if teql_with_penalty_final else 0,
                'median': np.median(teql_with_penalty_final) if teql_with_penalty_final else 0,
                'n': len(teql_with_penalty_final)
            }
        }
    
    plt.tight_layout()
    
    # 创建figures文件夹并保存
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/final_performance_analysis_ablation.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("Final performance ablation plot saved as 'figures/final_performance_analysis_ablation.pdf'")
    plt.show()
    
    return all_statistics

def create_statistical_tables(convergence_stats, performance_stats, analysis_type):
    """创建统计表格"""
    print(f"\n{'='*100}")
    print(f"{analysis_type.upper()} ANALYSIS STATISTICAL TABLES")
    print("="*100)
    
    if convergence_stats:
        print(f"\n{analysis_type} CONVERGENCE TIME ANALYSIS")
        print("-" * 80)
        
        for env, env_stats in convergence_stats.items():
            print(f"\n{env.upper()} ENVIRONMENT - Convergence Time")
            print("-" * 60)
            
            for threshold, threshold_stats in env_stats.items():
                print(f"\n{int(threshold*100)}% Performance Threshold:")
                print("-" * 40)
                
                df_data = []
                for method, values in threshold_stats.items():
                    if 'mean' in values:
                        df_data.append({
                            'Method': method,
                            'Mean': f"{values['mean']:.0f}",
                            'Std': f"{values['std']:.0f}",
                            'Median': f"{values['median']:.0f}",
                            'Sample Size': values['n']
                        })
                
                df = pd.DataFrame(df_data)
                print(df.to_string(index=False))
    
    if performance_stats:
        print(f"\n{analysis_type} FINAL PERFORMANCE ANALYSIS")
        print("-" * 80)
        
        for env, stats in performance_stats.items():
            print(f"\n{env.upper()} ENVIRONMENT - Final Performance")
            print("-" * 60)
            
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

def main():
    """主函数 - 运行所有分析"""
    print("Starting Comprehensive Analysis...")
    print("="*80)
    
    # 1. TLR vs TEQL 主要对比
    print("\n1. Running TLR vs TEQL Convergence Time Analysis...")
    convergence_main = plot_convergence_comparison_main()
    
    print("\n2. Running TLR vs TEQL Final Performance Analysis...")
    performance_main = plot_final_performance_main()
    
    # 2. TEQL 消融研究
    print("\n3. Running TEQL Convergence Time Ablation Study...")
    convergence_ablation = plot_convergence_comparison_ablation()
    
    print("\n4. Running TEQL Final Performance Ablation Study...")
    performance_ablation = plot_final_performance_ablation()
    
    # 3. 生成统计表格
    create_statistical_tables(convergence_main, performance_main, "TLR vs TEQL")
    create_statistical_tables(convergence_ablation, performance_ablation, "TEQL Ablation")
    
    print("\n" + "="*80)
    print("Comprehensive Analysis Completed!")
    print("All plots saved in 'figures/' folder as high-resolution PDFs.")
    print("="*80)

if __name__ == "__main__":
    main()