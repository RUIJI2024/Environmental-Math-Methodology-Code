# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:48:48 2024

@author: 吉芮
"""


# 牛顿插值
def newton_interpolation(x, y, value):
    """
   使用牛顿插值法来估计多项式的值
   参数:
        x (list): x坐标
        y (list): y坐标
        value (float): 想要估计多项式值的x值
    
    Returns:
        float: 给定x值处的估计y值
    """
    n = len(x)  # 数据点数量
    # Step 1: 创建一个2D列表来存储除差
    diff_table = [[0] * n for _ in range(n)]
    
    # Step 2: 用y值填充第一列
    for i in range(n):
        diff_table[i][0] = y[i]
    
    # Step 3:计算除差表的其余部分
    for j in range(1, n):  # 从第二列开始
        for i in range(n - j):  
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x[i + j] - x[i])
    
    # Step 4: 用差的除数来构造多项式
    result = diff_table[0][0]  # 从第一项开始
    product = 1  
    for i in range(1, n):
        product *= (value - x[i - 1])  # 每个k乘以（value - xk）
        result += diff_table[0][i] * product  # 加上多项式的下一项

    return result

# 示例数据点
x_points = [2, 3, -1, 1, -2]  #x坐标
y_points = [27, -33, 3, 21, -33]  # y坐标
value_to_estimate = 0  #想要估计y的x值

# 调用函数并输出结果
result = newton_interpolation(x_points, y_points, value_to_estimate)
print(f"The estimated value at x = {value_to_estimate} is: {result}")
