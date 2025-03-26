from sympy.strategies.core import switch

from matrix import *

# 这篇文章简单，玩法是直接给你上穷举

from itertools import combinations


def find_controller_place(matrix, k):
    """
    使用穷举法计算最优控制器放置方案

    参数：
        matrix: 图形邻接矩阵 (list of list)，表示节点间的时延。
        k: 控制器数量 (int)。
        v: 光传播速度 (float)，单位时延。

    返回：
        controller_place: 最优控制器位置组合 (list)。
        switch_set: 每个控制器管理的交换机集合 (list of list)。
    """
    n = len(matrix)  # 节点数量
    best_latency = float('inf')  # 初始化最优时延为无限大
    best_controller_place = []  # 初始化最优控制器位置
    best_switch_set = []  # 初始化最优交换机分配

    # Generate all possible combinations of k controllers
    controller_combinations = list(combinations(range(n), k))

    for controller_place in controller_combinations:
        # 根据当前控制器位置分配交换机
        switch_set = [[] for _ in range(k)]  # 初始化控制器的交换机集合
        for switch in range(n):  # 对每个交换机进行分配
            closest_controller = None
            closest_latency = float('inf')
            for i, controller in enumerate(controller_place):
                latency = matrix[switch][controller]  # 计算时延
                if latency < closest_latency:
                    closest_latency = latency
                    closest_controller = i
            switch_set[closest_controller].append(switch)

            # 计算当前组合的时延
        sc_avg, sc_worst, cc_avg, cc_worst = compute_latency(matrix, switch_set, controller_place)
        total_latency = sc_avg + cc_avg  # 综合时延指标（可调整权重）

        # 更新最优解
        if total_latency < best_latency:
            best_latency = total_latency
            best_controller_place = controller_place
            best_switch_set = switch_set

    return best_controller_place, best_switch_set

def compute_latency(matrix,switch_set,controller_place):
    # 传入参数包括matrix，管理交换机集合switch_set，控制器位置controller_place [3, 5 , 6 , 8]
    sc_worst_latency = 0
    sc_average_latency = 0
    cc_worst_latency = 0
    cc_average_latency = 0
    # 先计算控制器与交换机之间的平均时延与最大时延
    n = len(controller_place)
    for i in range(n):
        j = len(switch_set[i])
        for z in range(j):
            sc_worst_latency = max(sc_worst_latency,matrix[controller_place[i]][switch_set[i][z]])
            sc_average_latency += matrix[controller_place[i]][switch_set[i][z]]
    sc_average_latency = sc_average_latency/(n+len(matrix)) # 总时延除以平均时延
    for x in range(n):
        for y in range(n):
            if y > x:
                cc_worst_latency = max(cc_worst_latency,matrix[x][y])
                cc_average_latency += matrix[x][y]
    cc_average_latency = cc_average_latency/n
    return sc_average_latency,sc_worst_latency,cc_average_latency,cc_worst_latency

def load(switch_set,controller_place):
    # 传入参数为matrix, swtich_set   controller_place
    maxs = 0;mins = 999;n=0
    for switch in switch_set:
        n += len(switch)
        maxs = max(maxs,len(switch))
        mins = min(mins,len(switch))
    k = 1-(maxs-mins)/(n/len(controller_place))
    return k

def reliance():
    pass


if __name__ == "__main__":
    file_path="Iris.txt"
    # 光速 * 2/3
    c = 3 * 10 ** 5
    v = (c * 2) / 3
    matrix = matrix_gen(file_path,9999,v)
    print(find_controller_place(matrix, 5))