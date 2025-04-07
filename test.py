import itertools
from MinimizingLatencyforControllerPlacementProbleminSDN import mini_algorithm
from The_SDN_Controller_Placement_Problem_for_WAN import the_algorithm

def save_results_to_file_mini(filename, toponame, nodes_num, links_num, controller_num, avg_latency, \
                         worst_latency,cc_avg,cc_worst,load,avg_compute_time):
    '''

    :param filename: 要存入数据的文件名
    :return:
    '''
    toponame = toponame.replace(".txt","")
    with open(filename,'a') as f:
        f.write(f"mini_{toponame}_{controller_num}{{\n")
        f.write(f'    "toponame":"{toponame}"\n')
        f.write(f'    "nodes_num":{nodes_num}\n')
        f.write(f'    "links_num":{links_num}\n')
        f.write(f'    "controller_num":{controller_num}\n')
        f.write(f'    "avg_latency":{avg_latency}\n')
        f.write(f'    "worst_latency":{worst_latency}\n')
        f.write(f'    "cc_avg":{cc_avg}\n')
        f.write(f'    "cc_worst":{cc_worst}\n')
        f.write(f'    "load":{load}\n')
        f.write(f'    "avg_compute_time":{avg_compute_time}\n')
        f.write(f"}}\n")


def save_results_to_file_the(filename, toponame, nodes_num, links_num, controller_num, avg_latency, \
                         worst_latency,load,avg_compute_time):
    '''

    :param filename: 要存入数据的文件名
    :return:
    '''
    toponame = toponame.replace(".txt","")
    with open(filename,'a') as f:
        f.write(f"the_{toponame}_{controller_num}{{\n")
        f.write(f'    "toponame":"{toponame}"\n')
        f.write(f'    "nodes_num":{nodes_num}\n')
        f.write(f'    "links_num":{links_num}\n')
        f.write(f'    "controller_num":{controller_num}\n')
        f.write(f'    "avg_latency":{avg_latency}\n')
        f.write(f'    "worst_latency":{worst_latency}\n')
        f.write(f'    "load":{load}\n')
        f.write(f'    "avg_compute_time":{avg_compute_time}\n')
        f.write(f"}}\n")


def automated_test_mini(topologies, controller_counts, test_num, output_file="text.txt"):
    for topology,num in itertools.product(topologies, controller_counts):
        toponame,n,en,contronller_num,sc_average_latency,sc_worst_latency,cc_average_latency,\
        cc_worst_latency,loadd,run_time=mini_algorithm(topology,num,test_num)
        save_results_to_file_mini("test_data.txt",toponame,n,en,contronller_num,sc_average_latency,
                             sc_worst_latency,cc_average_latency,cc_worst_latency,loadd,run_time)


def automated_test_the(topologies, controller_counts, test_num, output_file="text.txt"):
    for topology,num in itertools.product(topologies, controller_counts):
        toponame,n,en,contronller_num,sc_average_latency,sc_worst_latency,\
        loadd,run_time=the_algorithm(topology,num,test_num)
        save_results_to_file_the("test_data.txt",toponame,n,en,contronller_num,sc_average_latency,
                             sc_worst_latency,loadd,run_time)


if __name__ == "__main__":
    #实现了自动化测试，只要修改下面的值就行
    topologies = ["Iris.txt"]
    contoller_num = [1,2,3,4,5,6]
    automated_test_mini(topologies,contoller_num,100)