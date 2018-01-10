# -*- coding: utf-8 -*-
"""
Created on  : 2018/1/8 10:05

@author: 张艳升
"""
'''
sklearn聚类算法实战
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture

'''
产生数据的函数
'''
def create_data(centers, num=100, std=0.7):
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return X, labels_true

def plot_data(*data):
    X, labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = 'rgbyckm'
    for i, label in enumerate(labels):
        position = labels_true==label
        ax.scatter(X[position, 0], X[position, 1], label="cluster %d"%label, color=colors[i%len(colors)])
    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()

# X, labels_true = create_data([[1,1],[2,2],[1,2],[10,20]], 1000, 0.5)
# plot_data(X, labels_true)

'''
K均值聚类KMeans
'''
def test_Kmeans(*data):
    X, lables_true = data
    clst = cluster.KMeans()
    clst.fit(X)
    predicted_labels = clst.predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true, predicted_labels))
    print("Sum center distance %s"%clst.inertia_)

# centers = [[1,1],[2,2],[1,2],[10,20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_Kmeans(X, labels_true)
'''
考察簇的数量的影响
'''
def test_Kmeans_nclusters(*data):
    X, labels_true = data
    nums = range(1, 50)
    ARIs = []
    Distances = []
    for num in nums:
        clst = cluster.KMeans(n_clusters=num)
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances.append(clst.inertia_)

    ##绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(nums, ARIs, marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("API")
    ax = fig.add_subplot(1,2,2)
    ax.plot(nums, Distances, marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("intertia_")
    fig.suptitle("KMeans")
    plt.show()

# centers = [[1,1],[2,2],[1,2],[10,20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_Kmeans_nclusters(X, labels_true)

'''
考察k均值算法运行的次数和选择初始中心向量策略的影响
'''


def test_Kmeans_n_init(*data):
    X, labels_true = data
    nums = range(1, 50)
    ##绘图
    fig = plt.figure()
    ARIs_k = []
    Distances_k = []
    ARIs_r = []
    Distances_r = []
    for num in nums:
        clst = cluster.KMeans(n_init=num, init='k-means++')
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs_k.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances_k.append(clst.inertia_)

        clst = cluster.KMeans(n_init=num, init='random')
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs_r.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances_r.append(clst.inertia_)

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs_k, marker="+", label="k-means++")
    ax.plot(nums, ARIs_r, marker="+", label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances_k, marker='o', label="k-means++")
    ax.plot(nums, Distances_r, marker='+', label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("intertia_")
    ax.legend(loc='best')

    fig.suptitle("KMeans")
    plt.show()

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_Kmeans_n_init(X, labels_true)

'''
使用密度聚类
'''
def test_DBSCAN(*data):
    X,labels_true = data
    clst = cluster.DBSCAN()
    #clst.fit(X)
    predicted_labels=clst.fit_predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))
    print("Core sample num: %d"%len(clst.core_sample_indices_))

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_DBSCAN(X, labels_true)

'''
考察epsilons参数的影响
'''
def test_DBSCAN_epsilon(*data):
    X,labels_true = data
    epsilons = np.logspace(-1,1.5)
    ARIs=[]
    Core_nums=[]
    for epsilon in epsilons:
        clst = cluster.DBSCAN(eps=epsilon)
        #clst.fit(X)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    #绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(epsilons,ARIs,marker="+")
    ax.set_xlabel("epsilons")
    ax.set_xscale('log')
    ax.set_ylim(0,1)
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(epsilons,Core_nums,marker='o')
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Core_Nums")
    ax.set_xscale('log')
    fig.suptitle("DBSCAN")
    plt.show()

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_DBSCAN_epsilon(X, labels_true)


'''
考察MinPts参数的影响
'''
def test_DBSCAN_min_samples(*data):
    X,labels_true = data
    min_samples=range(1,100)
    ARIs=[]
    Core_nums=[]
    for num in min_samples:
        clst = cluster.DBSCAN(min_samples=num)
        #clst.fit(X)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    #绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(min_samples,ARIs,marker="+")
    ax.set_xlabel("min_samples")
    ax.set_xscale('log')
    ax.set_ylim(0,1)
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(min_samples,Core_nums,marker='o')
    ax.set_xlabel("min_samples")
    ax.set_ylabel("Core_Nums")
    ax.set_xscale('log')
    fig.suptitle("DBSCAN")
    plt.show()

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_DBSCAN_min_samples(X, labels_true)


'''
层次聚类
'''
def test_AgglomerativeClustering(*data):
    X,labels_true = data
    clst = cluster.AgglomerativeClustering()
    #clst.fit(X)
    predicted_labels=clst.fit_predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))
   # print("Sum center distance %s"%clst.core_sample_indices_)

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_AgglomerativeClustering(X, labels_true)

'''
考察簇的数量对聚类效果的影响
'''
def test_AgglomerativeClustering_nclusters(*data):
    X,labels_true = data
    nums = range(1,50)
    ARIs=[]
    #Distances=[]
    for num in nums:
        clst = cluster.AgglomerativeClustering(n_clusters=num)
        #clst.fit(X)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
       # Distances.append(clst.inertia_)
    #绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("Agglomerativelustering")
    plt.show()

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_AgglomerativeClustering_nclusters(X, labels_true)

'''
考察链接方式的影响
'''
def test_AgglomerativeClustering_linkage(*data):
    X, labels_true = data
    nums = range(1, 50)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    linkages = ['ward', 'complete', 'average']
    markers = "+o*"
    for i, linkage in enumerate(linkages):
        ARIs = []
        for num in nums:
            clst = cluster.AgglomerativeClustering(n_clusters=num, linkage=linkage)
            # clst.fit(X)
            predicted_labels = clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker=markers[i], label="linkage:%s" % linkage)
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_AgglomerativeClustering_linkage(X, labels_true)

'''
混合高斯模型
'''
def test_GMM(*data):
    X,labels_true = data
    clst = mixture.GaussianMixture()
    clst.fit(X)
    predicted_labels=clst.predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_GMM(X, labels_true)

'''
考察簇的数量
'''
def test_GMM_n_components(*data):
    X,labels_true = data
    nums = range(1,50)
    ARIs=[]
    #Distances=[]
    for num in nums:
        clst = mixture.GaussianMixture(n_components=num)
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
       # Distances.append(clst.inertia_)
    #绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_components")
    ax.set_ylabel("ARI")
    fig.suptitle("GMM")
    plt.show()

# centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
# X, labels_true = create_data(centers, 1000, 0.5)
# test_GMM_n_components(X, labels_true)

'''
考察协方差类型的影响
'''
def test_GMM_cov_type(*data):
    X, labels_true = data
    nums = range(1, 50)

    cov_types = ['spherical', 'tied', 'diag', 'full']
    markers = "+o*s"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, cov_type in enumerate(cov_types):
        ARIs = []
        for num in nums:
            clst = mixture.GaussianMixture(n_components=num, covariance_type=cov_type)
            clst.fit(X)
            predicted_labels = clst.predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker=markers[i], label="covariance_type:%s" % cov_type)
    ax.set_xlabel("n_components")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("GMM")
    plt.show()

centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
X, labels_true = create_data(centers, 1000, 0.5)
test_GMM_cov_type(X, labels_true)
