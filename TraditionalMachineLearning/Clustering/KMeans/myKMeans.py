# -*- coding: utf-8 -*-
"""
Created on  : 2018/1/2 19:40

@author: 张艳升
"""
'''
实现自己的聚类
'''
# 导入numpy库
from numpy import *

# K-均值聚类辅助函数

# 文本数据解析函数
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每一行的数据映射成float型
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

# 数据向量计算欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 随机初始化K个质心(质心满足数据边界之内)
def randCent(dataSet, k):
    # 得到数据样本的维度
    n = shape(dataSet)[1]
    # 初始化为一个(k,n)的矩阵
    centroids = mat(zeros((k, n)))
    # 遍历数据集的每一维度
    for j in range(n):
        # 得到该列数据的最小值
        minJ = min(dataSet[:, j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    # 返回初始化得到的k个质心向量
    return centroids

# k-均值聚类算法
# @dataSet:聚类数据集
# @k:用户指定的k个类
# @distMeas:距离计算方法，默认欧氏距离distEclud()
# @createCent:获得k个质心的方法，默认随机获取randCent()
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据集样本数
    m = shape(dataSet)[0]
    # 初始化一个(m,2)的矩阵
    clusterAssment = mat(zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为false
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离最正无穷；最小距离对应索引为-1
            minDist = inf;
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)
                    minDist = distJI;
                    minIndex = j
        # 当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
        if clusterAssment[i, 0] != minIndex: clusterChanged = True
        # 更新当前变化样本的聚类结果和平方误差
        clusterAssment[i, :] = minIndex, minDist ** 2
    # 打印k-均值聚类的质心
    print(centroids)
    # 遍历每一个质心
    for cent in range(k):
        # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
        ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
        # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
        centroids[cent, :] = mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment

#二分K-均值聚类算法
#@dataSet:待聚类数据集
#@k：用户指定的聚类个数
#@distMeas:用户指定的距离计算方法，默认为欧式距离计算
def biKmeans(dataSet,k,distMeas=distEclud):
    #获得数据集的样本数
    m=shape(dataSet)[0]
    #初始化一个元素均值0的(m,2)矩阵
    clusterAssment=mat(zeros((m,2)))
    #获取数据集每一列数据的均值，组成一个长为列数的列表
    centroid0=mean(dataSet,axis=0).tolist()[0]
    #当前聚类列表为将数据集聚为一类
    centList=[centroid0]
    #遍历每个数据集样本
    for j in range(m):
        #计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    #循环，直至二分k-均值达到k类为止
    while (len(centList)<k):
        #将当前最小平方误差置为正无穷
        lowerSSE=inf
        #遍历当前每个聚类
        for i in range(len(centList)):
            #通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster=\
                dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat,splitClustAss=\
                kMeans(ptsInCurrCluster,2,distMeas)
            #计算该类划分后两个类的误差平方和
            sseSplit=sum(splitClustAss[:,1])
            #计算数据集中不属于该类的数据的误差平方和
            sseNotSplit=\
                sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #打印这两项误差值
            print('sseSplit,and notSplit:',(sseSplit,sseNotSplit))
            #划分第i类后总误差小于当前最小总误差
            if(sseSplit+sseNotSplit)<lowerSSE:
                #第i类作为本次划分类
                bestCentToSplit=i
                #第i类划分后得到的两个质心向量
                bestNewCents=centroidMat
                #复制第i类中数据点的聚类结果即误差值
                bestClustAss=splitClustAss.copy()
                #将划分第i类后的总误差作为当前最小误差
                lowerSSE=sseSplit+sseNotSplit
        #数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
        #当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=\
                len(centList)
        #同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        #连续不出现空缺
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=\
                bestCentToSplit
        #打印本次执行2-均值聚类算法的类
        print('the bestCentToSplit is:',bestCentToSplit)
        #打印被划分的类的数据个数
        print('the len of bestClustAss is:',(len(bestClustAss)))
        #更新质心列表中的变化后的质心向量
        centList[bestCentToSplit]=bestNewCents[0,:]
        #添加新的类的质心向量
        centList.append(bestNewCents[1,:])
        #更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:,0].A==\
                bestCentToSplit)[0],:]=bestClustAss
        #返回聚类结果
        return mat(centList),clusterAssment

#Yahoo！PlaceFinder API
#导入urllib
import urllib
#导入json模块
import json

#利用地名，城市获取位置经纬度函数
def geoGrab(stAddress,city):
    #获取经纬度网址
    apiStem='http://where.yahooapis.com/geocode?'
    #初始化一个字典，存储相关参数
    params={}
    #返回类型为json
    params['flags']='J'
    #参数appid
    params['appid']='ppp68N8t'
    #参数地址位置信息
    params['location']=('%s %s', (stAddress,city))
    #利用urlencode函数将字典转为URL可以传递的字符串格式
    url_params=urllib.urlencode(params)
    #组成完整的URL地址api
    yahooApi=apiStem+url_params
    #打印该URL地址
    print('%s',yahooApi)
    #打开URL，返回json格式的数据
    c=urllib.urlopen(yahooApi)
    #返回json解析后的数据字典
    return json.load(c.read())

from time import sleep
#具体文本数据批量地址经纬度获取函数
def massPlaceFind(fileName):
    #新建一个可写的文本文件，存储地址，城市，经纬度等信息
    fw=open('places.txt','wb+')
    #遍历文本的每一行
    for line in open(fileName).readlines():
        #去除首尾空格
        line =line.strip()
        #按tab键分隔开
        lineArr=line.split('\t')
        #利用获取经纬度函数获取该地址经纬度
        retDict=geoGrab(lineArr[1],lineArr[2])
        #如果错误编码为0，表示没有错误，获取到相应经纬度
        if retDict['ResultSet']['Error']==0:
            #从字典中获取经度
            lat=float(retDict['ResultSet']['Results'][0]['latitute'])
            #维度
            lng=float(retDict['ResultSet']['Results'][0]['longitute'])
            #打印地名及对应的经纬度信息
            print('%s\t%f\t%f',(lineArr[0],lat,lng))
            #将上面的信息存入新的文件中
            fw.write('%s\t%f\t%f\n',(line,lat,lng))
        #如果错误编码不为0，打印提示信息
        else:
            print('error fetching')
        #为防止频繁调用API，造成请求被封，使函数调用延迟一秒
        sleep(1)
    #文本写入关闭
    fw.close()

import math

#球面距离计算及簇绘图函数
def distSLC(vecA,vecB):
    #sin()和cos()以弧度未输入，将float角度数值转为弧度，即*pi/180
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*\
        cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return math.acos(a+b)*6371.0

import matplotlib
import matplotlib.pyplot as plt

#@numClust：聚类个数，默认为5
def clusterClubs(numClust=5):
    datList=[]
    #解析文本数据中的每一行中的数据特征值
    for line in open('places.txt').readlines():
        lineArr=line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[4])])
        datMat=mat(datList)
        #利用2-均值聚类算法进行聚类
        myCentroids,clusterAssing=biKmeans(datMat,numClust,\
            distMeas=distSLC)
        #对聚类结果进行绘图
        fig=plt.figure()
        rect=[0.1,0.1,0.8,0.8]
        scatterMarkers=['s','o','^','8','p',\
            'd','v','h','>','<']
        axprops=dict(xticks=[],ytick=[])
        ax0=fig.add_axes(rect,label='ax0',**axprops)
        imgP=plt.imread('Portland.png')
        ax0.imshow(imgP)
        ax1=fig.add_axes(rect,label='ax1',frameon=False)
        for i in range(numClust):
            ptsInCurrCluster=datMat[nonzero(clusterAssing[:,0].A==i)[0],:]
            markerStyle=scatterMarkers[i % len(scatterMarkers)]
            ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle,s=90)
        ax1.scatter(myCentroids[:,0].flatten().A[0],\
            myCentroids[:,1].flatten().A[0],marker='+',s=300)
        #绘制结果显示
        plt.show()

datMat = mat(loadDataSet('testSet.txt'))
print(kMeans(datMat,4))



