import numpy as np
import networkx as nx
from deap import base, creator, tools, algorithms
import random
import matrix as ma
# ====================== 输入参数初始化 ======================
#num_switches = 50  # 交换机数量
num_controllers = 3  # 控制器数量
population_size = 50  # 种群大小
max_generations = 100  # 最大迭代次数
failure_prob_per_unit = 0.01  # 单位距离故障概率
infinit = 99999

# 示例数据生成（实际场景需替换为真实数据）
# delay_matrix = np.random.rand(num_switches, num_switches)
# np.fill_diagonal(delay_matrix, 0)

# distance_matrix = np.random.randint(1, 10, (num_switches, num_switches))
# np.fill_diagonal(distance_matrix, 0)
distance_matrix, num_switches, num_links = ma.distance_matrix_gen("Iris.txt", infinit)
delay_matrix = ma.delay_matrix_gen(distance_matrix, infinit)

# 构建网络拓扑并计算最短路径
G = nx.Graph()
for i in range(num_switches):
    for j in range(i + 1, num_switches):
        if distance_matrix[i][j] > 0:
            G.add_edge(i, j, weight=distance_matrix[i][j])
shortest_distances = dict(nx.all_pairs_dijkstra_path_length(G))
print(shortest_distances)

# ====================== 修复重复控制器位置的函数 ======================
def repair_duplicates(individual):
    unique = list(set(individual))
    while len(unique) < num_controllers:
        candidates = [x for x in range(num_switches) if x not in unique]
        if not candidates:
            break
        unique.append(random.choice(candidates))
    random.shuffle(unique)
    individual[:] = unique[:num_controllers]
    return individual


# ====================== 多目标适应度函数 ======================
def evaluate(individual):
    controllers = individual
    if len(set(controllers)) != num_controllers:
        return (float('inf'), float('inf'), 0.0)  # 返回最差适应度
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

    return (avg_delay, load_std, avg_survival)


# ====================== NSGA-II 配置 ======================
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_switches), num_controllers)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 自定义交叉和变异操作，包含修复步骤
def cxTwoPointRepair(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    repair_duplicates(ind1)
    repair_duplicates(ind2)
    return ind1, ind2

def mutUniformIntRepair(individual):
    tools.mutUniformInt(individual, 0, num_switches-1, 0.1)
    repair_duplicates(individual)
    return individual,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_switches - 1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# ====================== 优化执行 ======================
population = toolbox.population(n=population_size)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg_delay", lambda pf: np.mean([ind[0] for ind in pf]))
stats.register("avg_load_std", lambda pf: np.mean([ind[1] for ind in pf]))
stats.register("avg_survival", lambda pf: np.mean([ind[2] for ind in pf]))

population, logbook = algorithms.eaMuPlusLambda(
    population, toolbox,
    mu=population_size,
    lambda_=population_size,
    cxpb=0.8,
    mutpb=0.2,
    ngen=max_generations,
    stats=stats,
    verbose=True
)

# ====================== 结果提取 ======================
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# 示例：选择综合最优解（加权法）
weights = [0.4, 0.3, -0.3]  # 权重需根据需求调整（第三目标为最大化，故取负）
best_solution = min(pareto_front,
                    key=lambda ind: sum(w * val for w, val in zip(weights, ind.fitness.values)))

print("最优控制器位置:", best_solution)
print(f"平均时延: {best_solution.fitness.values[0]:.3f}")
print(f"负载标准差: {best_solution.fitness.values[1]:.3f}")
print(f"平均存活概率: {best_solution.fitness.values[2]:.3f}")