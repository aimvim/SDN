import timeit
import time
import random
import numpy as np
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matrix as ma
import matrix_not_short as mas

# ====================== 输入参数初始化 ======================
num_controllers = 4  # 控制器数量
infinit = 99999
toponame = "Bics.txt"
test_num = 1

# 生成距离矩阵和延迟矩阵
distance_matrix, num_switches, num_links, edges, matrix = mas.distance_matrix_gen(toponame, infinit)
dis_matrix, _, _ = ma.distance_matrix_gen(toponame, infinit)
delay_matrix = ma.delay_matrix_gen(dis_matrix, infinit)

# 构建网络拓扑并计算最短路径
G = nx.Graph()
for i in range(num_switches):
    for j in range(i + 1, num_switches):
        if distance_matrix[i][j] < infinit:
            G.add_edge(i, j, weight=distance_matrix[i][j])
shortest_distances = dict(nx.all_pairs_dijkstra_path_length(G))


# ====================== 适应度函数 ======================
def evaluate(solution):
    controllers = solution
    delays = []
    assignments = []

    # 分配交换机到最近的控制器
    for switch in range(num_switches):
        closest = min(controllers, key=lambda c: delay_matrix[switch][c])
        delay = delay_matrix[switch][closest]
        delays.append(delay)
        assignments.append(closest)

    # 目标1: 平均时延
    avg_delay = np.mean(delays)
    worst_delay = max(delays)

    # 目标2: 负载均衡标准差
    load_counts = [assignments.count(c) for c in controllers]
    load_std = np.std(load_counts)

    # 归一化处理
    normalized_delay = avg_delay / 10.0  # 假设时延最大值约为10
    normalized_load = load_std / 5.0    # 假设负载标准差最大值约为5
    
    # 组合目标: 加权和方法平衡平均时延和负载均衡 (越小越好)
    w1, w2 = 0.5, 0.5  # 时延权重更高
    fitness = w1 * normalized_delay + w2 * normalized_load
    
    return fitness, avg_delay, load_std, worst_delay


# ====================== 辅助函数 ======================
def repair_solution(solution):
    """确保解中的控制器位置是唯一的，且数量正确"""
    # 确保值在有效范围内
    solution = [int(round(x)) % num_switches for x in solution]
    # 确保唯一性
    unique = list(set(solution))
    while len(unique) < num_controllers:
        candidates = [x for x in range(num_switches) if x not in unique]
        unique.append(random.choice(candidates))
    return sorted(unique[:num_controllers])


def early_stopping_check(fitness_history, patience=20, min_improvement=1e-6):
    """早停检测：如果连续patience代没有显著改善，则停止"""
    if len(fitness_history) < patience + 1:
        return False
    
    # 检查最近patience代的改善
    recent_best = min(fitness_history[-patience:])
    previous_best = min(fitness_history[:-patience])
    
    improvement = previous_best - recent_best
    return improvement < min_improvement


def elite_preservation(kites, elite_ratio=0.1):
    """精英保留策略：保留一定比例的最优个体"""
    num_elites = max(1, int(len(kites) * elite_ratio))
    # 按适应度排序
    kites.sort(key=lambda x: x['fitness'])
    # 返回精英个体
    return kites[:num_elites]


# ====================== 简化版黑翅鸢算法 ======================
def bka_optimization_simplified():
    # 算法参数
    num_kites = 50  # 黑翅鸢数量
    max_iterations = 150  # 最大迭代次数
    elite_ratio = 0.1  # 精英保留比例
    patience = 25  # 早停耐心值
    min_improvement = 1e-6  # 最小改善阈值
    
    # 初始化黑翅鸢种群
    kites = []
    for _ in range(num_kites):
        # 随机初始化控制器位置
        solution = repair_solution([random.randint(0, num_switches-1) for _ in range(num_controllers)])
        fitness, avg_delay, load_std, worst_delay = evaluate(solution)
        kites.append({
            'position': solution,
            'fitness': fitness,
            'avg_delay': avg_delay,
            'load_std': load_std,
            'worst_delay': worst_delay
        })
    
    # 按适应度排序
    kites.sort(key=lambda x: x['fitness'])
    
    # 初始化最优黑翅鸢
    best_kite = kites[0].copy()
    
    # 记录每代的最优适应度值，用于绘制收敛曲线
    fitness_history = []
    
    # 主循环
    for iteration in range(max_iterations):
        # 精英保留
        elites = elite_preservation(kites, elite_ratio)
        
        # 更新每只黑翅鸢的位置
        for i in range(num_kites):
            # 当前黑翅鸢
            current_kite = kites[i]
            new_position = current_kite['position'].copy()
            
            # 攻击行为 (Attacking behavior)
            # 根据BKA论文攻击行为公式(5)和(6)
            # 计算参数n：n = 0.05 × e^(-2×(t/T)²)
            n = 0.05 * np.exp(-2 * (iteration / max_iterations) ** 2)
            
            for j in range(num_controllers):
                # 生成随机数r
                r = random.random()  # 0到1之间的随机数
                p = 0.9  # 概率阈值
                
                # 根据公式(5)进行位置更新
                if p < r:
                    # y_{t+1}^{i,j} = y_t^{i,j} + n(1 + sin(r)) × y_t^{i,j}
                    new_pos = current_kite['position'][j] + n * (1 + np.sin(r)) * current_kite['position'][j]
                else:
                    # y_{t+1}^{i,j} = y_t^{i,j} + n × (2r - 1) × y_t^{i,j}
                    new_pos = current_kite['position'][j] + n * (2 * r - 1) * current_kite['position'][j]
                
                # 确保位置在有效范围内
                new_position[j] = max(0, min(num_switches - 1, int(round(new_pos))))
            
            # 修复解以确保有效性
            new_position = repair_solution(new_position)
            
            # 评估新位置
            fitness, avg_delay, load_std, worst_delay = evaluate(new_position)
            
            # 更新当前黑翅鸢的位置
            if fitness < kites[i]['fitness']:
                kites[i] = {
                    'position': new_position,
                    'fitness': fitness,
                    'avg_delay': avg_delay,
                    'load_std': load_std,
                    'worst_delay': worst_delay
                }
            
            # 迁徙行为 (Migration behavior)
            # 获取当前黑翅鸢的适应度
            Fi = current_kite['fitness']
            # 随机选择一个黑翅鸢
            rand_kite_idx = random.randint(0, num_kites - 1)
            rand_kite = kites[rand_kite_idx]
            Fri = rand_kite['fitness']
            
            new_position = current_kite['position'].copy()
            
            # 根据BKA论文迁徙公式(7)和(8)
            for j in range(num_controllers):
                # 计算随机参数r
                r = random.random()
                
                # 计算m参数：m = 2 × sin(r + π/2)
                m = 2 * np.sin(r + np.pi / 2)
                
                # 获取领导者位置（最优黑翅鸢位置）
                L_j = best_kite['position'][j]
                
                # 根据公式(7)和(8)进行位置更新
                if Fi < Fri:
                    # y_{t+1}^{i,j} = y_t^{i,j} + C(0,1) × (y_t^{i,j} - L_j)
                    cauchy_random = np.random.standard_cauchy()  # 柯西分布随机数C(0,1)
                    new_pos = current_kite['position'][j] + cauchy_random * (current_kite['position'][j] - L_j)
                else:
                    # y_{t+1}^{i,j} = y_t^{i,j} + C(0,1) × (L_j - m × y_t^{i,j})
                    cauchy_random = np.random.standard_cauchy()  # 柯西分布随机数C(0,1)
                    new_pos = current_kite['position'][j] + cauchy_random * (L_j - m * current_kite['position'][j])
                
                # 确保位置在有效范围内
                new_position[j] = max(0, min(num_switches - 1, int(round(new_pos))))
            
            # 修复解以确保有效性
            new_position = repair_solution(new_position)
            
            # 评估新位置
            fitness, avg_delay, load_std, worst_delay = evaluate(new_position)
            
            # 更新当前黑翅鸢的位置
            if fitness < kites[i]['fitness']:
                kites[i] = {
                    'position': new_position,
                    'fitness': fitness,
                    'avg_delay': avg_delay,
                    'load_std': load_std,
                    'worst_delay': worst_delay
                }
        
        # 精英保留：确保精英个体不会丢失
        kites.sort(key=lambda x: x['fitness'])
        for i, elite in enumerate(elites):
            if i < len(kites) and elite['fitness'] < kites[-(i+1)]['fitness']:
                kites[-(i+1)] = elite
        
        # 重新排序黑翅鸢
        kites.sort(key=lambda x: x['fitness'])
        
        # 更新最优解
        if kites[0]['fitness'] < best_kite['fitness']:
            best_kite = kites[0].copy()

        # 记录当前代的最优适应度
        fitness_history.append(best_kite['fitness'])
        
        # 早停检测
        if early_stopping_check(fitness_history, patience, min_improvement):
            print(f"早停触发：第 {iteration} 代，连续 {patience} 代无显著改善")
            break
        
        # 每10代输出进度
        if iteration % 10 == 0:
            print(f"第 {iteration} 代，最佳适应度: {best_kite['fitness']:.6f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_history)), fitness_history, 'b-', linewidth=2, label='Best Fitness')
    plt.title('Simplified BKA Algorithm Convergence Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('bka_03_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回最优解和收敛历史
    return best_kite, fitness_history


# ====================== 运行优化 ======================
print(f"开始使用简化版黑翅鸢算法求解SDN控制器放置问题...")
print(f"拓扑: {toponame}, 节点数: {num_switches}, 控制器数: {num_controllers}")
print(f"优化策略: 早停检测 + 精英保留")

# 记录开始时间
start_time = time.time()

# 运行优化算法
best_solution, fitness_history = bka_optimization_simplified()

# 计算算法运行时间
runtime = time.time() - start_time
iterations_used = len(fitness_history)

# 输出结果
print("\n=== 简化版BKA算法优化结果 ===")
print(f"控制器位置: {best_solution['position']}")
print(f"平均时延: {best_solution['avg_delay']:.6f}")
print(f"负载标准差: {best_solution['load_std']:.6f}")
print(f"最差时延: {best_solution['worst_delay']:.6f}")
print(f"综合适应度: {best_solution['fitness']:.6f}")
print(f"实际迭代次数: {iterations_used}")
print(f"算法运行时间: {runtime:.4f} 秒")
print(f"收敛曲线已保存为 'bka_03_convergence.png'")

# 保存结果到文件
result_file = os.path.join(os.getcwd(), 'bka_03_results.txt')
with open(result_file, 'a') as f:
    f.write(f"=== Simplified BKA Algorithm Results ===\n")
    f.write(f"toponame: {toponame}, num_switches: {num_switches}, num_controllers: {num_controllers}\n")
    f.write(f"position: {best_solution['position']}\n")
    f.write(f"avg_delay: {best_solution['avg_delay']:.6f}\n")
    f.write(f"load_std: {best_solution['load_std']:.6f}\n")
    f.write(f"worst_delay: {best_solution['worst_delay']:.6f}\n")
    f.write(f"fitness: {best_solution['fitness']:.6f}\n")
    f.write(f"iterations_used: {iterations_used}\n")
    f.write(f"run_time: {runtime:.4f} s\n")
    f.write(f"收敛曲线已保存为 'bka_03_convergence.png'\n")
    f.write(f"优化策略: 早停检测 + 精英保留\n")
    f.write("\n" + "="*50 + "\n\n")

print(f"结果已保存到 '{result_file}'")