# -*- coding: utf-8 -*-
"""
Created on  : 2017/12/19 8:53

@author: 张艳升
"""
'''
简单逻辑回归的实现
'''
import numpy as np
import random
'''
简易版
随机梯度下降算法
X，样本
y，样本标签
theta，待学习参数
alpha，学习率
m，样本个数
numIteration，迭代次数
'''
def gradientDescent(X, y, theta, alpha, m, numIteration):
    XTrans = X.transpose()
    for i in range(0, numIteration):
        hypothesis = np.dot(X, theta) #输出函数，即sigmoid函数
        loss = hypothesis - y #损失，即误差error
        cost = np.sum(loss**2)/(2*m) #计算损失函数
        print("Iteration %d | Cost: %f" % (i, cost))#打印每次迭代的损失
        gradient = np.dot(XTrans, loss)/m #梯度，即类似于求导
        theta = theta - alpha * gradient #核心随机梯度下降
    return theta
'''
简易版获取数据函数
此函数可以换成自己的获取数据的函数
numPoints，样本个数
bias，偏向
variance，方差
'''
def genData(numPoints, bias, variance):
    X = np.zeros(shape=(numPoints, 2)) #样本，行数numPoints，列数2
    y = np.zeros(shape=numPoints) #样本标签，行数numPoints，列数1
    for i in range(0, numPoints): #赋值循环
        X[i][0] = 1
        X[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return X, y
'''
参数可以调节，来达到最优化
'''
x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)
