# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 09:31:40 2024

@author: 吉芮
"""

"""Frame for the task using monte carlo:
    monte carlo simulation 概率=积分面积
    首先在x（0-pi），y（0-1）【y的区间由给定函数和定义域决定】中生成N个随机数
    其次 得出（n）选取出的y中有多少是小于sin（x）的
    monte carlo simulation n/N 概率=积分面积
    """
    
import numpy as np
# use numpy for caculate sin(x) and generate random numbers N

def f(x):
    return np.sin(x)
#use it for next comparation

def monte_carlo_simution(N):
    x=np.random.uniform(0,np.pi,N)
    y=np.random.uniform(0,1,N)
    n=np.sum(y<f(x))
    ratio=n/N
    #概率=积分面积
    return ratio

N=10000000
#更改N的数值得到更贴近的值
result=monte_carlo_simution(N)
print(result)


    
    