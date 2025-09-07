# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:55:34 2024

@author: 吉芮 环境工程 22012031
"""

import numpy as np
from scipy.stats import rankdata

#数据
data = np.array([5.1, 5.3, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.4, 6.7, 6.9, 7.1, 7.5, 7.6, 7.8, 8.0, 8.3, 8.6, 8.7])

# 步骤一: 计算数据的秩
ranks = rankdata(data)

# 步骤二：计算pettitt 的K 值
n = len(data)
#计算数据长度
U_t = np.array([2 * np.sum(ranks[:t]) - t * (n + 1) for t in range(1, n)])
K = np.max(np.abs(U_t))
K_index = np.argmax(np.abs(U_t)) + 1  # Adjust index because U_t starts from t=1

#步骤三：计算P值
p_value = 2 * np.exp((-6 * K**2) / (n**3 + n**2))


print(K)
print(K_index)
#发生突变的数据位置
print(p_value)
#检验突变是否是随机的
