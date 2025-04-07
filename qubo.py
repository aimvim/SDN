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
qubo_model.set_objective(kw.qubo.quicksum([delay_matrix[i][j]*x[i,j] for i in range(n) for j in range(n)]))

#约束每个交换机由且仅由一个控制器管辖
qubo_model.add_constraint(x.sum(axis=1) == 1,"switch_cons", penalty=10)

def check_controller(x):
    count = 0
    for col_sum in x.sum(axis=0):
        if col_sum != 0:
            count += 1
    return count

#约束控制器的数量为k
qubo_model.add_constraint(check_controller(x)==k,"controller_cons",penalty=10)

qubo_mat = qubo_model.get_qubo_matrix()
pd.DataFrame(qubo_mat).to_csv("cpp.csv",index=False,header=False)
