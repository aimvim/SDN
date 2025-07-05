def binary_list_to_matrix(binary_list, row_length):
    """
    将二进制列表转换为方阵
    :param binary_list: 二进制列表，包含0和1
    :param row_length: 每行的长度，即节点数
    :return: 方阵
    """
    if len(binary_list) % row_length != 0:
        raise ValueError("列表长度必须是每行长度的整数倍")

    matrix = [binary_list[i:i + row_length] for i in range(0, len(binary_list), row_length)]
    return matrix


def find_controllers_and_switches(matrix):
    """
    找出控制器的位置及其管辖的交换机位置
    :param matrix: 方阵
    :return: 控制器位置及其管辖的交换机位置
    """
    controllers = []
    for j in range(len(matrix[0])):
        controller = {
            "controller_position": j,
            "managed_switches": []
        }
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                controller["managed_switches"].append(i)
        if controller["managed_switches"]:
            controllers.append(controller)
    return controllers


def main():
    # 输入二进制列表和每行的长度
    binary_list = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    row_length = 4

    # 将二进制列表转换为方阵
    matrix = binary_list_to_matrix(binary_list, row_length)
    print("方阵：")
    for row in matrix:
        print(row)

    # 找出控制器的位置及其管辖的交换机位置
    controllers = find_controllers_and_switches(matrix)
    print("\n控制器位置及其管辖的交换机位置：")
    for controller in controllers:
        print(f"控制器位置: {controller['controller_position']}, 管辖的交换机位置: {controller['managed_switches']}")


if __name__ == "__main__":
    main()