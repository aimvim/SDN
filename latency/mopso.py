import random
import timeit

import numpy as np
import networkx as nx
from deap import base, creator, tools
import matrix_not_short as mas
import matrix as ma

# ====================== 初始化 DEAP 个体结构 ======================
# 定义多目标适应度（时延、负载标准差最小化，存活概率最大化）
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# ====================== 输入参数初始化 ======================
#num_switches = 50  # 交换机数量
num_controllers = 6  # 控制器数量
failure_prob_per_unit = 0.01  # 单位距离故障概率
infinit = 99999
toponame = "Netrail.txt"
test_num = 100

distance_matrix, num_switches, num_links, edges, matrix = mas.distance_matrix_gen(toponame, infinit)
dis_matrix, _, _ = ma.distance_matrix_gen(toponame, infinit)
delay_matrix = ma.delay_matrix_gen(dis_matrix, infinit)

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
        if distance_matrix[i][j] < infinit:
            G.add_edge(i, j, weight=distance_matrix[i][j])
shortest_distances = dict(nx.all_pairs_dijkstra_path_length(G))


# ====================== 多目标适应度函数 ======================
def evaluate(solution):
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
    worst_delay = max(delays)

    # 目标2: 负载均衡标准差
    load_counts = [assignments.count(c) for c in controllers]
    load_std = np.std(load_counts)

    # 目标3: 平均存活概率
    avg_survival = np.mean(survival_probs)

    return (avg_delay, load_std, avg_survival),worst_delay


# ====================== 支配关系判断函数 ======================
def is_dominated(fitness_a, fitness_b):
    """判断 fitness_a 是否被 fitness_b 支配（假设目标为最小化时延、负载标准差，最大化存活概率）"""
    # 条件1: fitness_b 在至少一个目标上严格优于 fitness_a
    cond_any_better = (
            (fitness_b[0] < fitness_a[0]) or  # 时延更小
            (fitness_b[1] < fitness_a[1]) or  # 负载标准差更小
            (fitness_b[2] > fitness_a[2])  # 存活概率更大
    )
    # 条件2: fitness_b 在所有目标上不差于 fitness_a
    cond_all_not_worse = (
            (fitness_b[0] <= fitness_a[0]) and
            (fitness_b[1] <= fitness_a[1]) and
            (fitness_b[2] >= fitness_a[2])
    )
    return cond_all_not_worse and cond_any_better


# ====================== MOPSO 核心算法 ======================
def mopso_optimization():
    # 超参数
    num_particles = 50
    max_iterations = 200
    inertia_weight = 0.4
    cognitive_weight = 1.0
    social_weight = 1.0
    archive_size = 50
    mutation_prob = 0.1

    # 辅助函数：修正位置唯一性
    def repair_position(position):
        position = [int(round(x)) % num_switches for x in position]
        unique = list(set(position))
        while len(unique) < num_controllers:
            candidates = [x for x in range(num_switches) if x not in unique]
            unique.append(random.choice(candidates))
        return sorted(unique[:num_controllers])

    # 初始化粒子群和存档
    particles = []
    archive = []  # 外部存档（存储帕累托解，使用 DEAP 个体对象）

    for _ in range(num_particles):
        # 创建 DEAP 个体
        position = repair_position([random.uniform(0, num_switches) for _ in range(num_controllers)])
        ind = creator.Individual(position)
        ind.fitness.values,worst_delay = evaluate(ind)
        particles.append({
            'position': position,
            'velocity': [random.uniform(-1, 1) for _ in range(num_controllers)],
            'best_position': position.copy(),
            'best_fitness': ind.fitness.values
        })
        # 更新存档（非支配解）
        if not any(is_dominated(ind.fitness.values, a.fitness.values) for a in archive):
            # 移除被当前解支配的旧解
            archive = [a for a in archive if not is_dominated(a.fitness.values, ind.fitness.values)]
            archive.append(ind)

    # 主循环
    for _ in range(max_iterations):
        # 修剪存档（使用 DEAP 的 sortNondominated）
        if len(archive) > archive_size:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            tools.emo.assignCrowdingDist(fronts[0])
            crowd_dist = [ind.fitness.crowding_dist for ind in fronts[0]]
            selected_indices = np.argsort(crowd_dist)[-archive_size:]
            archive = [fronts[0][i] for i in selected_indices]

        # 更新每个粒子
        for p in particles:
            # 选择全局引导者（从存档随机选一个非支配解）
            leader = random.choice(archive).copy() if archive else creator.Individual(p['best_position'])

            # 更新速度
            new_velocity = [
                inertia_weight * v +
                cognitive_weight * random.random() * (p_best - x) +
                social_weight * random.random() * (leader[i] - x)
                for i, (v, x, p_best) in enumerate(zip(
                    p['velocity'],
                    p['position'],
                    p['best_position']
                ))
            ]

            # 更新位置（连续空间 -> 离散修正）
            new_position = [x + v for x, v in zip(p['position'], new_velocity)]
            new_position = repair_position(new_position)

            # 变异操作
            if random.random() < mutation_prob:
                idx = random.randint(0, num_controllers - 1)
                new_position[idx] = random.choice([x for x in range(num_switches) if x not in new_position])

            # 计算新解适应度
            new_ind = creator.Individual(new_position)
            new_ind.fitness.values,worst_delay = evaluate(new_ind)

            # 更新个体最优（若新解非支配）
            if not is_dominated(new_ind.fitness.values, p['best_fitness']):
                p['best_position'] = new_position.copy()
            p['best_fitness'] = new_ind.fitness.values

            # 更新全局存档
            if all(not is_dominated(new_ind.fitness.values, a.fitness.values) for a in archive):
                # 移除被新解支配的旧解
                archive = [a for a in archive if not is_dominated(a.fitness.values, new_ind.fitness.values)]
            archive.append(new_ind)

            # 保存新状态
            p['position'] = new_position
            p['velocity'] = new_velocity

            # 提取帕累托前沿
    return archive,worst_delay


def save_results_to_file_mopso(filename, toponame, nodes_num, links_num, controller_num, avg_latency,
                         worst_latency,load,avg_compute_time):
    '''

    :param filename: 要存入数据的文件名
    :return:
    '''
    toponame = toponame.replace(".txt","")
    with open(filename,'a') as f:
        f.write(f"mopso_{toponame}_{controller_num}{{\n")
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
pareto_front, worst_delay = mopso_optimization()

#计算算法运行时间
runtime = timeit.timeit(mopso_optimization,number=test_num)/test_num


# 示例：选择拥挤度最大的解作为代表性解
fronts = tools.sortNondominated(pareto_front, len(pareto_front), first_front_only=True)
tools.emo.assignCrowdingDist(fronts[0])
crowd_dist = [ind.fitness.crowding_dist for ind in fronts[0]]
best_idx = np.argmax(crowd_dist)
best_solution = fronts[0][best_idx]

print("帕累托前沿解数量:", len(pareto_front))
print("代表性解控制器位置:", sorted(best_solution))
print(f"平均时延: {best_solution.fitness.values[0]}")
print(f"负载标准差: {best_solution.fitness.values[1]}")
print(f"平均存活概率: {best_solution.fitness.values[2]}")
# save_results_to_file_mopso("..\\test_data.txt", toponame, num_switches, num_links,
#                            num_controllers, best_solution.fitness.values[0],worst_delay,
#                            best_solution.fitness.values[1],runtime)