import math
import re
import itertools


def distance_compute(lat1, lon1, lat2, lon2):
    """  
    计算两地经纬度之间的距离 (单位: 公里).  

    参数:  
    - lat1, lon1: 第一点的纬度和经度  
    - lat2, lon2: 第二点的纬度和经度  

    返回值:  
    - 距离 (单位: 公里)  
    """
    # 将度数转化为弧度  
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 地球的平均半径（单位: 公里）  
    R = 6371.0

    # Haversine 公式  
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算距离  
    distance = R * c
    return distance


def distance_matrix_gen(file_path,infinit):
    """
    将文本文件解析为距离矩阵，其中两点之间的距离使用 distance_compute 函数计算。

    参数:
    - file_path: 存储节点信息的输入文件路径。

    返回值:
    - 距离矩阵 (二维列表)
    """
    # 存储每个节点的位置信息 {id: (lat, lon)}
    node_positions = {}
    edges = []
    # 读取文件并提取节点信息
    with open(file_path, "r") as file:
        data = file.read()

        # 使用正则表达式找到所有节点信息块
    node_blocks = re.findall(r'node\s*\[\s*(.*?)\s*\]', data, re.DOTALL)

    for block in node_blocks:
        # 提取 id, Latitude, Longitude
        node_id = re.search(r'id\s+(\d+)', block)
        latitude = re.search(r'Latitude\s+([-\d.]+)', block)
        longitude = re.search(r'Longitude\s+([-\d.]+)', block)

        if node_id and latitude and longitude:
            node_id = int(node_id.group(1))  # 转为整数 id
            lat = float(latitude.group(1))  # 纬度
            lon = float(longitude.group(1))  # 经度
            node_positions[node_id] = (lat, lon)

    edge_blocks = re.findall(r'edge\s*\[\s*(.*?)\s*\]', data, re.DOTALL)
    for block in edge_blocks:
        # 提取 source, target 节点
        source = re.search(r'source\s+(\d+)', block)
        target = re.search(r'target\s+(\d+)', block)
        if source and target:
            source_id = int(source.group(1))
            target_id = int(target.group(1))
            edges.append((source_id, target_id))  # 添加边信息
    # nodes_positions记录了全部的节点，edges记录了全部的边
    node_ids = sorted(node_positions.keys());n = len(node_ids);en=len(edges);print(n);print(en)
    # 构造初始矩阵
    dis_matrix = [[infinit for _ in range(n)] for _ in range(n)]
    # 填入初始数值
    for i in range(en):
        dis_matrix[edges[i][0]][edges[i][1]] = (distance_compute(node_positions[edges[i][0]][0],node_positions[edges[i][0]][1],node_positions[edges[i][1]][0],node_positions[edges[i][1]][1]))
        dis_matrix[edges[i][1]][edges[i][0]] = dis_matrix[edges[i][0]][edges[i][1]]
    # 接下来计算每条边的最短路径
    for z in range(n):
        for x in range(n):
            for y in range(n):
                if x == y:
                    dis_matrix[x][y] = 0.0
                dis_matrix[x][y] = min(dis_matrix[x][y],dis_matrix[x][z]+dis_matrix[z][y])
    return dis_matrix,n,en

def delay_matrix_gen(dis_matrix, infinite):
    """
        将距离转换成延迟，参考了GitHub上的转换代码，考虑将非真空条件下的光速设置为
        c=1.97 * 10**8 m/s
        t = distance / speed of light
        t (in ms) = ( distance in km * 1000 (for meters) ) / ( speed of light / 1000 (for ms))

        参数:
        - dis_matrix: 存储节点间距离的矩阵。

        返回值:
        - 延迟矩阵
    """
    c=1.97 * 10**8
    delay_matrix = [[element*1000/(c/1000) if element != infinite else element for element in row] for row in dis_matrix]
    return delay_matrix


def compute_failed_link(k,distance_matrix,edges,infinit):
    '''
    遍历了k条失效链路的所有情况算出平均时延
    :param k:链路失效数
    :param distance_matrix:未最短路径的距离矩阵
    :param edges: 存边信息的列表
    :param infinit: 无穷
    :return: 一个k链路失效下所有组合的平均时延列表
    '''

    n = len(distance_matrix)
    fail_avg_delays = []
    all_possible_failures = list(itertools.combinations(edges, k))
    for failed_edges in all_possible_failures:
        # 创建一个距离矩阵的副本，用于当前失效情况
        current_dis_matrix = [row[:] for row in distance_matrix]

        # 设置失效链路的距离为无穷大
        for edge in failed_edges:
            current_dis_matrix[edge[0]][edge[1]] = infinit
            current_dis_matrix[edge[1]][edge[0]] = infinit

        # 重新计算最短路径
        for z in range(n):
            for x in range(n):
                for y in range(n):
                    if x == y:
                        current_dis_matrix[x][y] = 0.0
                    current_dis_matrix[x][y] = min(current_dis_matrix[x][y],
                                                   current_dis_matrix[x][z] + current_dis_matrix[z][y])

        #计算该情况下的平均时延
        delay_matrix = delay_matrix_gen(distance_matrix,infinit)
        sum_delay = 0
        for i in range(n):
            for j in range(i + 1, n):  # 只遍历 i < j 的位置
                if delay_matrix[i][j] != infinit:
                    sum_delay += delay_matrix[i][j]
        fail_avg_delays.append(sum_delay/n)


    return fail_avg_delays


def compute_connectivity_matrix(adj_matrix):
    n = len(adj_matrix)
    # 初始化连通矩阵，初始时如果两个节点之间有边，则为1，否则为0
    connectivity_matrix = [[1 if adj_matrix[i][j] == 1 else 0 for j in range(n)] for i in range(n)]

    # Floyd-Warshall 算法的核心部分
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # 如果 i 到 k 和 k 到 j 都连通，则 i 到 j 也连通
                connectivity_matrix[i][j] = connectivity_matrix[i][j] or (
                            connectivity_matrix[i][k] and connectivity_matrix[k][j])

    return connectivity_matrix


def compute_ncpr(adj_matrix, controller_place):
    """
    计算网络的连通性比例（NCPR）。

    参数:
    - adj_matrix: 邻接矩阵
    - controller_place: 控制器的位置列表

    返回:
    - NCPR: 网络的连通性比例
    """
    n = len(adj_matrix)
    con_matrix = compute_connectivity_matrix(adj_matrix)
    connected_switches = 0

    for i in range(n):
        # 检查当前节点是否能连接到所有控制器
        if all(con_matrix[i][j] for j in controller_place):
            connected_switches += 1

    ncpr = connected_switches / n
    return ncpr


# 测试函数
if __name__ == "__main__":
    # 假设输入文件是 "nodes.txt"
    infinite = 99999
    # Ntt 47个节点，216条边(有一大堆边重合),先别用
    # 下面几个也是有重合边的，说实话我不知道这个怎么去处理，写算法吧
    # Bics 33节点， 48条边
    # Arnes 34节点，47条边
    # Ntelos 48节点，61条边
    # Bellcanada 48节点 64条边
    # Iris 51节点，64条边
    file_path = "Bics.txt"
    c = 3 * 10 ** 5
    v = 1.97 * 10 ** 5
    matrix = distance_matrix_gen(file_path,infinite)
    print(matrix)
    for row in matrix:
        print(row)
