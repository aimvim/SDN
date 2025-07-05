import numpy as np
import kaiwu as kw
from matrix import *
import pandas as pd

# 初始化许可证
kw.license.init(user_id="92068964368154626", sdk_code="bJpy8WPQQUGE17XpJBo4u3BGJL4B6f")

# 参数设置
infinite = 9999
toponame = "ruan"
file_path = toponame + ".txt"
k = 2

# 生成距离矩阵和延迟矩阵
dis_matrix, n, en = distance_matrix_gen(file_path, infinite)
delay_matrix = delay_matrix_gen(dis_matrix, infinite)

# 创建 QUBO 变量矩阵
x = kw.qubo.ndarray((n, n), "x", kw.qubo.Binary)
y = kw.qubo.ndarray(n,"y", kw.qubo.Binary)  # yi表示第i个节点是否有控制器
qubo_model = kw.qubo.QuboModel()


# 设置目标函数：平均延迟
obj1 = kw.qubo.quicksum([delay_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)])

# 设置目标函数：负载均衡
L = [kw.qubo.quicksum([x[i, j] * y[j] for i in range(n)]) for j in range(n)]
avg_L = kw.qubo.quicksum(L) / k
obj2 = kw.qubo.quicksum([(L[j] - avg_L) ** 2 for j in range(n)])

# 设置权重
obj = obj1 + obj2
qubo_model.set_objective(obj)
kw.qubo.details(obj)

# 约束：每个交换机由且仅由一个控制器管辖，行和为1
qubo_model.add_constraint(x.sum(axis=1) == 1, "switch_cons", penalty=100, constr_type='hard')

# 约束：x与y之间的关系
for j in range(n):
    # 添加约束 y_j >= x_i,j 对于所有 i
    for i in range(n):
        qubo_model.add_constraint(y[j] >= x[i, j], f"y_{j}_geq_x_{i}_{j}_constraint", penalty=10, constr_type='hard')

    # 添加约束 y_j <= sum(x_i,j) 对于所有 i
    column_sum = kw.qubo.quicksum([x[i, j] for i in range(n)])
    qubo_model.add_constraint(y[j] <= column_sum, f"y_{j}_leq_sum_x_{j}_constraint", penalty=10, constr_type='hard')

#约束控制器数量为k
qubo_model.add_constraint(y.sum() == k, "controller_cons", penalty=50, constr_type='hard')


# 生成 QUBO 矩阵并求解
qubo_mat = qubo_model.get_qubo_matrix(8)
# np.fill_diagonal(qubo_mat, 0)
# mutated_mat = kw.preprocess.perform_precision_adaption_mutate(qubo_mat)
# pd.DataFrame(qubo_mat).to_csv("qubo_matrix.csv",index=False,header=False)

solver = kw.solver.SimpleSolver(kw.classical.SimulatedAnnealingOptimizer(initial_temperature=200,
                                                                         alpha=0.99,
                                                                         cutoff_temperature=0.001,
                                                                         iterations_per_t=50,
                                                                         size_limit=100))


# # 配置求解器
# save_dir = "C:\\code\\课题一\\SDN\\SDN\\results"  # 指定保存目录
# optimizer = kw.cim.CIMOptimizer(user_id="92068964368154626",sdk_code="bJpy8WPQQUGE17XpJBo4u3BGJL4B6f", task_name="test_task",save_dir=save_dir)
# optimizer = kw.preprocess.PrecisionReducer(optimizer, 8)  # 8位精度
# solver = kw.solver.SimpleSolver(optimizer)
sol_dict, qubo_val = solver.solve_qubo(qubo_model)

# 验证约束条件并计算评分
unsatisfied_count, res_dict = qubo_model.verify_constraint(sol_dict)
print("未满足的约束数量：", unsatisfied_count)
print("约束项的值：", res_dict)
print("解字典为：", sol_dict)
score_val = kw.qubo.get_val(qubo_model.objective, sol_dict)
print('评分为: {}'.format(score_val))