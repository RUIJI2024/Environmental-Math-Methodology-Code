# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:45:09 2024

@author: RUI JI
"""
import numpy as np
import matplotlib.pyplot as plt

#示例数据
X=np.array([4,5,6,7,8,10]).flatten()
Y=np.array([9.1,11.5,12.9,16,14,21]).flatten()

#自变量和因变量的平均值
x_mean=np.mean(X)
y_mean=np.mean(Y)

n=len(X)

#计算斜率
a_numerator=np.sum((X-x_mean)*(Y-y_mean))#分子
a_denominator=np.sum((X-x_mean)**2)#分母
a=a_numerator/a_denominator

#截距
b=y_mean-a*x_mean

#回归方程
y_predicted=a*X+b

#输出斜率和截距
print(f"slope:{a}")
print(f"intecept:{b}")

#相关系数r
r_numerator=np.sum((X-x_mean)*(Y-y_mean))
r_denominator=np.sqrt(np.sum((X-x_mean)**2)*np.sum((Y-y_mean)**2))
r=r_numerator/r_denominator
print(f"correlation coefficient r:{r}")

#t检验
t_value=r*np.sqrt((n-2)/(1-r**2))
print(f"t-value: {t_value}")

#t检验p值的计算 自由度为4 p单边显著性为0.05根据对照表 t为2.132
if t_value>2.132:
    print("可靠")
else:
    print("不可靠")



plt.scatter(X,Y,color='blue',label='original_date')
plt.plot(X,y_predicted,color='red',label='liner regression')
plt.xlabel('X(independt_variable)')
plt.ylabel('Y(dependt_variable)')
plt.legend()
plt.show()


""" pyhton 库检查

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Example data
X = np.array([4, 5, 6, 7, 8, 10]).reshape(-1, 1)
Y = np.array([9.1, 11.5, 12.9, 16, 14, 21])

# Using scikit-learn for linear regression
model = LinearRegression()
model.fit(X, Y)

# Predicted Y values
y_pred = model.predict(X)

# Slope and Intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (scikit-learn): {slope}")
print(f"Intercept (scikit-learn): {intercept}")

# R-squared value to check the goodness of fit
r2 = r2_score(Y, y_pred)
print(f"R-squared: {r2}")

# Mean Squared Error (MSE) to check the accuracy
mse = mean_squared_error(Y, y_pred)
print(f"Mean Squared Error: {mse}")

# Using scipy to calculate t-statistics and p-value for the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), Y)

print(f"Slope (scipy): {slope}")
print(f"Intercept (scipy): {intercept}")
print(f"Correlation coefficient r: {r_value}")
print(f"P-value: {p_value}")
print(f"Standard error: {std_err}")

# Plotting
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X, y_pred, color='red', label='Linear Regression (scikit-learn)')
plt.xlabel('X (Independent Variable)')
plt.ylabel('Y (Dependent Variable)')
plt.legend()
plt.show()

# Check if the p-value is less than 0.05 for statistical significance
if p_value < 0.05:
    print("The model is statistically significant.")
else:
    print("The model is not statistically significant.")
    """