# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:42:20 2024

@author: 86187
"""

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = b_curr = 0  # 当前的m和b值, 从零开始
    iterations = 10000 # 迭代次数, 需要走多少步, 后续可以调整找到最适合的
    n = len(x)
    learning_rate = 0.01  # m进行迭代的参数, 需要测试调整
    tolerance = 1e-8      # 停止迭代的阈值(-3,-8)
    """如果希望更精确，可以将阈值设得更小（如 1e-8）；
    如果希望更快停止，可以设得更大（如 1e-4）"""
    
    # 用于存储每次迭代的cost值，以便绘制图表
    costs = []
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr  # y的预测值
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        costs.append(cost)
        
        # 计算m和b的导数
        md = -(2/n) * sum(x * (y - y_predicted))  # m的导数
        bd = -(2/n) * sum(y - y_predicted)        # b的导数
        
        # 更新m和b的值
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        # 打印每次迭代的m, b, cost, iteration的值
        print(f"Iteration {i+1}: m = {m_curr:.4f}, b = {b_curr:.4f}, cost = {cost:.4f}")
        
        # 检查代价函数的变化是否小于阈值
        if i > 0 and abs(costs[-1] - costs[-2]) < tolerance:
            print(f"Converged at iteration {i+1}")
            break
    """abs(costs[-1] - costs[-2]) 计算当前代价函数与上一次代价函数的差值的绝对值。
        如果这个差值小于 tolerance，则认为代价函数的变化已经足够小，
        算法已经收敛，可以提前停止迭代。"""
    # 绘制最终的拟合结果和cost变化图表
    plt.figure(figsize=(16, 6))
    
    # 绘制拟合结果
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_predicted, color='red', label='Final Fitted Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Final Fitted Line: m = {m_curr:.4f}, b = {b_curr:.4f}')
    plt.legend()
    plt.grid(True)
    
    # 绘制cost随迭代次数变化的图表
    plt.subplot(1, 2, 2)
    plt.plot(range(len(costs)), costs, label='Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# 需要进行拟合的数组数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)