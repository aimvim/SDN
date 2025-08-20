import random
import timeit

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
    w1, w2 = 0.7, 0.3  # 时延权重更高
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


def local_search(solution):
    """对当前解进行局部搜索，尝试找到更好的解"""
    best_solution = solution.copy()
    best_fitness, _, _, _ = evaluate(best_solution)
    
    # 尝试替换每个控制器位置
    for i in range(len(solution)):
        current = solution[i]
        # 尝试替换为其他可能的位置
        for new_pos in range(num_switches):
            if new_pos not in solution:
                # 创建新解
                new_solution = solution.copy()
                new_solution[i] = new_pos
                new_solution = sorted(new_solution)  # 保持有序
                
                # 评估新解
                new_fitness, _, _, _ = evaluate(new_solution)
                
                # 如果找到更好的解，更新最佳解
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
    
    return best_solution


# ====================== 灰狼优化算法 ======================
def gwo_optimization():
    # 算法参数
    num_wolves = 50  # 增加灰狼数量以提高搜索多样性
    max_iterations = 150  # 增加迭代次数以获得更好收敛
    a_decrease_rate = 2 / max_iterations  # 参数a的线性递减率
    local_search_prob = 0.1  # 局部搜索概率
    
    # 初始化灰狼种群
    wolves = []
    for _ in range(num_wolves):
        # 随机初始化控制器位置
        solution = repair_solution([random.randint(0, num_switches-1) for _ in range(num_controllers)])
        fitness, avg_delay, load_std, worst_delay = evaluate(solution)
        wolves.append({
            'position': solution,
            'fitness': fitness,
            'avg_delay': avg_delay,
            'load_std': load_std,
            'worst_delay': worst_delay
        })
    
    # 按适应度排序
    wolves.sort(key=lambda x: x['fitness'])
    
    # 初始化Alpha, Beta, Delta狼
    alpha = wolves[0]
    beta = wolves[1] if len(wolves) > 1 else alpha
    delta = wolves[2] if len(wolves) > 2 else beta
    
    # 历史最优解记录，实现精英保留策略
    historical_best = alpha.copy()
    
    # 收敛检测变量
    convergence_threshold = 10  # 连续多少代无改进则认为收敛
    no_improvement_count = 0
    best_fitness = alpha['fitness']
    
    # 记录每代的最优适应度值，用于绘制收敛曲线
    fitness_history = []
    
    # 主循环
    for iteration in range(max_iterations):
        # 更新参数a (从2线性递减到0)
        a = 2 - iteration * a_decrease_rate
        
        # 自适应调整局部搜索概率
        current_local_search_prob = local_search_prob * (1 - iteration / max_iterations)
        
        # 更新每只灰狼的位置
        for i in range(num_wolves):
            # 对每个维度更新位置
            new_position = []
            for j in range(num_controllers):
                # 计算Alpha, Beta, Delta对当前狼的影响
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                # 计算Alpha, Beta, Delta引导的位置
                if j < len(alpha['position']) and j < len(beta['position']) and j < len(delta['position']):
                    X1 = alpha['position'][j] - A1 * abs(C1 * alpha['position'][j] - wolves[i]['position'][j])
                    X2 = beta['position'][j] - A2 * abs(C2 * beta['position'][j] - wolves[i]['position'][j])
                    X3 = delta['position'][j] - A3 * abs(C3 * delta['position'][j] - wolves[i]['position'][j])
                    
                    # 计算新位置
                    new_pos = (X1 + X2 + X3) / 3
                    new_position.append(new_pos)
                else:
                    # 如果索引超出范围，使用随机值
                    new_position.append(random.randint(0, num_switches-1))
            
            # 修复解以确保有效性
            new_position = repair_solution(new_position)
            
            # 局部搜索策略
            if random.random() < current_local_search_prob:
                # 尝试局部搜索改进解
                improved_position = local_search(new_position)
                improved_fitness, improved_avg_delay, improved_load_std, improved_worst_delay = evaluate(improved_position)
                
                if improved_fitness < fitness:
                    new_position = improved_position
                    fitness, avg_delay, load_std, worst_delay = improved_fitness, improved_avg_delay, improved_load_std, improved_worst_delay
            else:
                # 评估新位置
                fitness, avg_delay, load_std, worst_delay = evaluate(new_position)
            
            # 更新当前狼的位置
            if fitness < wolves[i]['fitness']:
                wolves[i] = {
                    'position': new_position,
                    'fitness': fitness,
                    'avg_delay': avg_delay,
                    'load_std': load_std,
                    'worst_delay': worst_delay
                }
        
        # 更新Alpha、Beta和Delta狼
        wolves.sort(key=lambda x: x['fitness'])
        alpha = wolves[0]
        beta = wolves[1] if len(wolves) > 1 else alpha
        delta = wolves[2] if len(wolves) > 2 else beta
        
        # 更新历史最优解
        if alpha['fitness'] < historical_best['fitness']:
            historical_best = alpha.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # 精英保留策略：确保历史最优解始终在种群中
        if historical_best['fitness'] < wolves[-1]['fitness']:
            wolves[-1] = historical_best.copy()
        
        # 收敛检测已在更新历史最优解时处理
        best_fitness = min(best_fitness, alpha['fitness'])
        
        # 早停条件
        if no_improvement_count >= convergence_threshold and iteration > max_iterations // 4:
            print(f"算法在第 {iteration} 代收敛，提前停止")
            break
            
        # 多样性保持策略
        if no_improvement_count >= convergence_threshold // 2:
            # 重新初始化部分狼群以增加多样性
            for i in range(num_wolves // 4, num_wolves // 2):
                # 保留前1/4的精英解，重新初始化后1/4到1/2的解
                new_solution = repair_solution([random.randint(0, num_switches-1) for _ in range(num_controllers)])
                fitness, avg_delay, load_std, worst_delay = evaluate(new_solution)
                wolves[i] = {
                    'position': new_solution,
                    'fitness': fitness,
                    'avg_delay': avg_delay,
                    'load_std': load_std,
                    'worst_delay': worst_delay
                }
            # 重置收敛计数器
            no_improvement_count = 0
        
        # 记录当前代的最优适应度
        fitness_history.append(alpha['fitness'])
        
        # 每10代输出进度
        if iteration % 10 == 0:
            print(f"第 {iteration} 代，最佳适应度: {alpha['fitness']}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_history)), fitness_history, 'b-', linewidth=2)
    plt.title('灰狼优化算法收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.grid(True)
    plt.savefig('gwo_convergence.png')
    plt.close()
    
    # 返回最优解和收敛历史
    return alpha, fitness_history


def save_results_to_file_gwo(filename, toponame, nodes_num, links_num, controller_num, avg_latency,
                         worst_latency, load, avg_compute_time):
    '''
    保存结果到文件
    :param filename: 要存入数据的文件名
    '''
    toponame = toponame.replace(".txt", "")
    with open(filename, 'a') as f:
        f.write(f"gwo_{toponame}_{controller_num}{{\n")
        f.write(f'    "toponame":"{toponame}"\n')
        f.write(f'    "nodes_num":{nodes_num}\n')
        f.write(f'    "links_num":{links_num}\n')
        f.write(f'    "controller_num":{controller_num}\n')
        f.write(f'    "avg_latency":{avg_latency}\n')
        f.write(f'    "worst_latency":{worst_latency}\n')
        f.write(f'    "load":{load}\n')
        f.write(f'    "avg_compute_time":{avg_compute_time}\n')
        f.write(f"}}\n")


# ====================== 运行优化 ======================
print(f"开始使用灰狼优化算法求解SDN控制器放置问题...")
print(f"拓扑: {toponame}, 节点数: {num_switches}, 控制器数: {num_controllers}")

# 记录开始时间
start_time = time.time()

# 运行优化算法
best_solution, fitness_history = gwo_optimization()

# 计算算法运行时间
runtime = time.time() - start_time

# 输出结果
print("\n最优解:")
print(f"控制器位置: {best_solution['position']}")
print(f"平均时延: {best_solution['avg_delay']}")
print(f"负载标准差: {best_solution['load_std']}")
print(f"最差时延: {best_solution['worst_delay']}")
print(f"综合适应度: {best_solution['fitness']}")
print(f"平均运行时间: {runtime} 秒")
print(f"收敛曲线已保存为 'gwo_convergence.png'")


# 保存结果到文件
# save_results_to_file_gwo("..\\test_data.txt", toponame, num_switches, num_links,
#                        num_controllers, best_solution['avg_delay'], best_solution['worst_delay'],
#                        best_solution['load_std'], runtime)