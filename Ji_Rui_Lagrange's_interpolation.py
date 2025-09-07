# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:16:49 2024

@author: 86187
"""
#定义方程
def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)  # 计算长度
    result = 0  # 初始结果

    for i in range(n):
        #y（i）
        term = y_points[i]

        # 计算li（x）
        for j in range(n):
            if j != i:  # 跳过选定的点
            #得出第i项的方程
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])

        # 相加求和
        result += term

    return result

# 给出数据点
x_points = [2, 3, -1, 1, -2]
y_points = [27, -33, 3, 21, -33]

# 测试计算x=0处的插值
x_value = 0
result = lagrange_interpolation(x_points, y_points, x_value)

# 输出结果
print(f"拉格朗日插值在x = {x_value} 处的结果是 {result}")


    
    
    
    