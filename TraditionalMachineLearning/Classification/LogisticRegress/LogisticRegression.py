# -*- coding: utf-8 -*-
"""
Created on  : 2017/12/18 14:36

@author: 张艳升
"""
'''
自己实现逻辑回归函数
用到loadDataSet()加载数据集
sigmoid()激活函数
'''
import numpy as np
import matplotlib.pyplot as plt

'''
加载数据集函数loadDataSet()
'''
def loadDataSet():
    dataMat = [] #样本特征
    labelMat = [] #类别标记
    fr = open('E:/MachineLearning/MachineLearning/Data/testSet.txt') #加载数据集
    for line in fr.readlines(): #依次读取一行
        lineArr = line.strip().split() #去除空格并分隔
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

'''
sigmoid()函数
'''
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

'''
梯度上升优化算法
来更新权重（参数）
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() #转置，列向量
    m, n = np.shape(dataMatrix) #行数、 列数
    alpha = 0.01 #学习步长，也称学习率
    maxCycles = 500 #迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) #h 为列向量
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

'''
随机梯度上升算法
'''
def SGDAscent(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix) #行数
    alpha = 0.01 #学习步长
    weights = np.ones(n) #权重
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))#这里与梯度上升算法不同，这里是一个数值
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
继续改进随机梯度上升算法
一方面alpha在每次迭代的时候都会调整
另一面随机选取样本来更新回归系数，即weights
'''
def SGDAscentImprove(dataMatrix, classLabels, numIter = 150):
    m, n = np.shape(dataMatrix)#行数
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01 #alpha每次更新时都需要调整
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            '''
            随机梯度上升是真实值减去预测值
            而随机梯度下降是预测值减去真实值
            '''
            error = classLabels[randIndex] - h
            weights = weights +alpha * error * dataMatrix[randIndex] #随机梯度下降就是weight-alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])#进行下一次迭代时删除该值
    return weights


'''
画出决策边界
'''
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] #行
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
'''
用逻辑回归来从疝气病症预测病马的死亡率
'''
'''
预测类别函数
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
'''
用来打开训练集与测试集，并对数据进行格式化处理的函数
'''
def colicTest():
    frTrain = open('E:/MachineLearning/MachineLearning/Data/horseColicTraining.txt')
    frTest = open('E:/MachineLearning/MachineLearning/Data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = SGDAscentImprove(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainingWeights)) != int(currLine[21]): #预测在这一句
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f"%errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f"%(numTests, errorSum/float(numTests)))

multiTest()

# dataArr, labelMat = loadDataSet()
# print(SGDAscent(np.array(dataArr), labelMat))
# print(SGDAscentImprove(np.array(dataArr),labelMat))
# #weights = gradAscent(dataArr, labelMat)
# #print(plotBestFit(weights))

