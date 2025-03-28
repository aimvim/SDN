import numpy as np
import timeit
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
        total_latency = 0.95*sc_avg + 0.05*cc_avg  # 综合时延指标（可调整权重）,这个位置的具体计算方法文章也没说，后面调一下权重就行

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
    sc_average_latency = sc_average_latency/(len(matrix)) # 总时延除以平均时延
    for x in controller_place:
        for y in controller_place:
            if y > x:
                cc_worst_latency = max(cc_worst_latency,matrix[x][y])
                cc_average_latency += matrix[x][y]
    cc_average_latency = cc_average_latency/n
    return sc_average_latency,sc_worst_latency,cc_average_latency,cc_worst_latency

def load(switch_set):
    # 传入参数为matrix, swtich_set   controller_place
    #maxs = 0;mins = 999;n=0
    controller_count = []
    for switch in switch_set:
    #     n += len(switch)
    #     maxs = max(maxs,len(switch))
    #     mins = min(mins,len(switch))
    # k = 1-(maxs-mins)/(n/len(controller_place))
        controller_count.append(len(switch))

    return np.std(controller_count)

def reliance():
    pass


def mini_algorithm(toponame, contronller_num,test_num):
    dis_matrix,n,en = distance_matrix_gen(toponame, 9999)
    delay_matrix = delay_matrix_gen(dis_matrix, 9999)
    controller_place, switch_set = find_controller_place(delay_matrix, contronller_num)
    run_time = timeit.timeit(lambda: find_controller_place(delay_matrix, contronller_num), number=test_num)
    run_time /= test_num
    sc_average_latency, sc_worst_latency, cc_average_latency, cc_worst_latency=\
        compute_latency(delay_matrix, switch_set, controller_place)
    loadd=load(switch_set)
    return toponame,n,en,contronller_num,sc_average_latency,sc_worst_latency,cc_average_latency,cc_worst_latency,loadd,run_time


# if __name__ == "__main__":
#     file_path="Bellcanada.txt"
#     # 光速 * 2/3
#     c = 3 * 10 ** 5
#     v = 1.97 * 10 ** 8
#     k=6
#     dis_matrix = distance_matrix_gen(file_path,9999)
#     delay_matrix = delay_matrix_gen(dis_matrix,9999)
#     controller_place,switch_set = find_controller_place(delay_matrix,k)
#     print(controller_place, switch_set)
#     run_time = timeit.timeit(lambda :find_controller_place(delay_matrix,k),number=10)
#     print(compute_latency(delay_matrix, switch_set, controller_place))
#     print(f"负载为：{load(switch_set)}")
#     print(f"平均计算时间为：{run_time/10}")
#     # 计算find_*** 100遍就行（拓扑   k的值）