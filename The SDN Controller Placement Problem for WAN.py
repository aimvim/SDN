from matrix import distance_compute
import re
import numpy as np
import random
# 这个的方法有点点复杂，我是看gpt看懂的
# 但是不知道gpt是否正确
# 但是————反正文章代码没开源，我随便复现
# 处理上没什么问题，但是复现后效果不是很好

def matrix_gen(file_path,infinit,v):
    # 这个比matrix那个还要多一步的是，这个要构造邻接矩阵
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
    node_ids = sorted(node_positions.keys());n = len(node_ids);en=len(edges)#print(n);print(en)
    # 构造初始矩阵
    dis_matrix = [[infinit for _ in range(n)] for _ in range(n)]
    # 构造邻接矩阵
    W = [[0 for _ in range(n)] for _ in range(n)]
    # 填入初始数值
    for i in range(en):
        dis_matrix[edges[i][0]][edges[i][1]] = (distance_compute(node_positions[edges[i][0]][0],node_positions[edges[i][0]][1],node_positions[edges[i][1]][0],node_positions[edges[i][1]][1]))/v * 1000
        dis_matrix[edges[i][1]][edges[i][0]] = dis_matrix[edges[i][0]][edges[i][1]]
        W[edges[i][0]][edges[i][1]] = 1; W[edges[i][1]][edges[i][0]] = 1
    # 接下来计算每条边的最短路径
    for z in range(n):
        for x in range(n):
            for y in range(n):
                if x == y:
                    dis_matrix[x][y] = 0.0
                dis_matrix[x][y] = min(dis_matrix[x][y],dis_matrix[x][z]+dis_matrix[z][y])
    return dis_matrix,W


def get_first_k_columns(L, k):
    """
    从矩阵 L 中提取前 k 列。

    参数:
        L:  二维数组 (列表的列表形式)，表示输入矩阵。
        k:  整数，表示要提取的列数。

    返回:
        result: 一个含有前 k 列的二维数组。
    """
    # 提取前 k 列
    result = [row[:k] for row in L]
    return result


def K_Cluster(Y, k, max_iter=100, tol=1e-64):
    """
    使用 k-均值对嵌入点进行聚类

    参数:
        Y: 原生列表形式的 n*k 矩阵，表示 n 个节点在 k 维空间的嵌入表示
        k: int，目标簇的数量
        max_iter: int，可选，最大迭代次数
        tol: float，可选，容忍误差，用于判断中心点是否收敛

    返回:
        clusters: list，每个簇的索引集合
        centers: np.ndarray，表示最终的聚类中心
    """
    # 将原生列表转为 NumPy 数组，以支持矢量化计算
    Y = np.array(Y)  # 转为 NumPy 数组
    n, d = Y.shape  # 矩阵的行数（点数）和列数（维度）

    # Step 1: 初始化聚类中心
    indices = random.sample(range(n), k)  # 随机选取 k 个点的索引
    centers = Y[indices]  # 提取对应的 k 个中心点

    # 确保 prev_centers 初始化为对应的 NumPy 数组
    prev_centers = np.zeros_like(centers)  # 初始化为 NumPy 数组（零矩阵）
    clusters = [[] for _ in range(k)]      # 用于存储每个簇的点索引

    for iteration in range(max_iter):
        # Step 2: 分配点到最近的聚类中心
        clusters = [[] for _ in range(k)]  # 清空每个簇的点

        for i in range(n):
            # 计算点 i 到每个聚类中心的距离
            distances = np.linalg.norm(Y[i] - centers, axis=1)  # 欧几里得距离
            # 分配到最近的聚类中心
            closest_center = np.argmin(distances)
            clusters[closest_center].append(i)

        # Step 3: 更新每个簇的聚类中心
        prev_centers = np.copy(centers)  # 深拷贝当前中心（确保是 NumPy 数组）
        for j in range(k):
            if clusters[j]:  # 如果簇不为空
                # 计算簇内所有点的均值作为新的中心
                centers[j] = np.mean(Y[clusters[j]], axis=0)

        # Step 4: 判断中心点是否收敛（变化小于容忍误差）
        if np.linalg.norm(centers - prev_centers) < tol:
            print(f"聚类在第 {iteration + 1} 次迭代后收敛")
            break

    return clusters, centers

def Spectral_Clustering(A,W,k):
    # 这里的输入是——距离矩阵S,邻接矩阵，控制器数量k
    n = len(A)#生成一个有多少节点
    degree_matrix = [[0 for _ in range(n)] for _ in range(n)] #构造度矩阵
    L = [[0 for _ in range(n)] for _ in range(n)] #构建拉普拉斯矩阵
    for i in range(n):
        for j in range(n):
            # 完成度矩阵的计算
            degree_matrix[i][i] += A[i][j]
    # 计算拉普拉斯L矩阵 = 度矩阵 - 邻接矩阵
    for x in range(n):
        for y in range(n):
            L[x][y] = degree_matrix[x][y] - W[x][y]
    Y = get_first_k_columns(L,k) # 取出拉普拉斯矩阵的前k列向量
    # 接下来就是对Y（包括y_1,y_2,...,y_n进行分类计算了）
    clusters,centers = K_Cluster(Y,k)
    return clusters

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
    file_path = ("Netrail.txt")
    c = 3 * 10 ** 5
    v = (c * 2) / 3
    matrix,W = matrix_gen(file_path, 9999, v)
    print(Spectral_Clustering(matrix,W,3))
