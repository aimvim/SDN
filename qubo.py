import numpy as np
import  pandas as pd
from matrix import *
import kaiwu as kw
kw.license.init(user_id="92068964368154626", sdk_code="bJpy8WPQQUGE17XpJBo4u3BGJL4B6f")

infinite=9999
toponame="Iris"
file_path=toponame+".txt"
k=4
dis_matrix,n,en=distance_matrix_gen(file_path,infinite)
delay_matrix=delay_matrix_gen(dis_matrix,infinite)

# Create qubo variable matrix
# x(i,j)表示第i个交换机与第j个节点上的控制器之间的管辖关系，值为1或0
x = kw.qubo.ndarray((n, n), "x", kw.qubo.Binary)
qubo_model = kw.qubo.QuboModel()

#设置目标函数，负载均衡
L = [kw.qubo.quicksum([x[i, j] for i in range(n)]) for j in range(k)]
avg_L = kw.qubo.quicksum(L) / k
obj2 = kw.qubo.quicksum([(L[j] - avg_L) ** 2 for j in range(k)])
#设置目标函数，平均延迟
obj1=kw.qubo.quicksum([delay_matrix[i][j]*x[i,j] for i in range(n) for j in range(n)])
#设置权重
obj=0.5*obj1+0.5*obj2
qubo_model.set_objective(obj)

#自动确定惩罚系数
# row_sums = x.sum(axis=1)
# cons1 = kw.qubo.quicksum([(row_sums[i] - 1)**2 for i in range(n)])
# cons2 = (kw.qubo.quicksum([x[j, i] for i in range(n) for j in range(n)]) - k)**2
# penalty1 = kw.qubo.get_min_penalty(obj1, cons1)
# penalty2 = kw.qubo.get_min_penalty(obj1, cons2)
# print(penalty1,penalty2)

#约束每个交换机由且仅由一个控制器管辖
qubo_model.add_constraint(x.sum(axis=1) == 1,"switch_cons", penalty=10,constr_type='hard')

#约束控制器的数量为k
controller_sum = kw.qubo.quicksum([x[j, i] for i in range(n) for j in range(n)])
qubo_model.add_constraint(controller_sum == k,"controller_cons",penalty=10,constr_type='hard')

qubo_mat = qubo_model.get_qubo_matrix()
np.fill_diagonal(qubo_mat, 0)
mutated_mat = kw.cim.adjust_ising_matrix_precision(qubo_mat)
pd.DataFrame(mutated_mat).to_csv("qubo_matrix.csv",index=False,header=False)

# 报错：The maximum number of bits supported by the license: 100, Please log in to platform.qboson.com for more bit support.
# #Perform calculation using SA optimizer
# solver = kw.solver.SimpleSolver(kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,
#                                                                          alpha=0.99,
#                                                                          cutoff_temperature=0.001,
#                                                                          iterations_per_t=10,
#                                                                          size_limit=100))
# sol_dict, qubo_val = solver.solve_qubo(qubo_model)
#
# # Check the hard constraints for validity and path length
# unsatisfied_count, res_dict = qubo_model.verify_constraint(sol_dict)
# print("unsatisfied constraint: ", unsatisfied_count)
# print("value of constraint term", res_dict)
#
# # Calculate the score
# score_val = kw.qubo.get_val(qubo_model.objective, sol_dict)
# print('评分为: {}'.format(score_val))