import math
import re

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


def matrix_gen(file_path,infinit,v):
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
        dis_matrix[edges[i][0]][edges[i][1]] = (distance_compute(node_positions[edges[i][0]][0],node_positions[edges[i][0]][1],node_positions[edges[i][1]][0],node_positions[edges[i][1]][1]))/v * 1000
        dis_matrix[edges[i][1]][edges[i][0]] = dis_matrix[edges[i][0]][edges[i][1]]
    # 接下来计算每条边的最短路径
    for z in range(n):
        for x in range(n):
            for y in range(n):
                if x == y:
                    dis_matrix[x][y] = 0.0
                dis_matrix[x][y] = min(dis_matrix[x][y],dis_matrix[x][z]+dis_matrix[z][y])
    return dis_matrix
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
    v = (c * 2) / 3
    matrix = matrix_gen(file_path,infinite,v)
    print(matrix)
    for row in matrix:
        print(row)
