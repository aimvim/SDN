import numpy as np
import random
import math
import networkx as nx
import matrix as ma

# ====================== 输入参数初始化 ======================
#num_switches = 50  # 交换机数量
num_controllers = 3  # 控制器数量
failure_prob_per_unit = 0.01  # 单位距离故障概率
infinit = 99999

distance_matrix, num_switches, num_links = ma.distance_matrix_gen("Iris.txt", infinit)
delay_matrix = ma.delay_matrix_gen(distance_matrix, infinit)

# # 示例数据生成（实际场景需替换为真实数据）
# delay_matrix = np.random.rand(num_switches, num_switches)
# np.fill_diagonal(delay_matrix, 0)
#
# distance_matrix = np.random.randint(1, 10, (num_switches, num_switches))
# np.fill_diagonal(distance_matrix, 0)

# 构建网络拓扑并计算最短路径
G = nx.Graph()
for i in range(num_switches):
    for j in range(i + 1, num_switches):
        if distance_matrix[i][j] > 0:
            G.add_edge(i, j, weight=distance_matrix[i][j])
shortest_distances = dict(nx.all_pairs_dijkstra_path_length(G))


# ====================== 目标函数 ======================
def calculate_score(solution, weights):
    controllers = solution
    delays, loads, survival_probs = [], [], []
    assignments = []

    # 分配交换机到最近的控制器
    for switch in range(num_switches):
        closest = min(controllers, key=lambda c: delay_matrix[switch][c])
        delay = delay_matrix[switch][closest]
        delays.append(delay)
        assignments.append(closest)

        # 计算链路存活概率
        d = shortest_distances[switch][closest] if switch != closest else 0
        survival = (1 - failure_prob_per_unit) ** d
        survival_probs.append(survival)

    # 目标1: 平均时延
    avg_delay = np.mean(delays)

    # 目标2: 负载均衡标准差
    load_counts = [assignments.count(c) for c in controllers]
    load_std = np.std(load_counts)

    # 目标3: 平均存活概率
    avg_survival = np.mean(survival_probs)

    # 综合评分（加权求和）
    score = weights[0] * avg_delay + weights[1] * load_std - weights[2] * avg_survival
    return score, avg_delay, load_std, avg_survival


# ====================== 模拟退火算法 ======================
def simulated_annealing():
    # 超参数
    initial_temp = 1000
    cooling_rate = 0.95
    max_iterations = 1000
    weights = [0.4, 0.3, 0.3]  # 时延、负载、稳定性权重

    # 生成初始解
    current_solution = random.sample(range(num_switches), num_controllers)
    current_score, avg_delay, load_std, survival = calculate_score(current_solution, weights)

    best_solution = current_solution.copy()
    best_score = current_score
    best_metrics = (avg_delay, load_std, survival)

    temp = initial_temp

    for i in range(max_iterations):
        # 生成邻域解（随机替换一个控制器）
        new_solution = current_solution.copy()
        replace_idx = random.randint(0, num_controllers - 1)
        new_controller = random.choice([x for x in range(num_switches) if x not in new_solution])
        new_solution[replace_idx] = new_controller

        # 计算新解评分
        new_score, new_avg, new_load, new_survival = calculate_score(new_solution, weights)

        # 接受准则
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_solution = new_solution
            current_score = new_score
            if new_score < best_score:
                best_solution = new_solution.copy()
                best_score = new_score
                best_metrics = (new_avg, new_load, new_survival)

        # 降温
        temp *= cooling_rate

    return best_solution, best_score, best_metrics


# ====================== 运行优化 ======================
best_solution, best_score, (avg_delay, load_std, survival) = simulated_annealing()

# 输出结果
print("最优控制器位置:", sorted(best_solution))
print(f"综合评分: {best_score:.3f}")
print(f"平均时延: {avg_delay:.3f}")
print(f"负载标准差: {load_std:.3f}")
print(f"平均存活概率: {survival:.3f}")