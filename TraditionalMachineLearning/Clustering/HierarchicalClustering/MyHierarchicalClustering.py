# -*- coding: utf-8 -*-
"""
Created on  : 2018/1/6 16:07

@author: 张艳升
"""
'''
实现自己的层次聚类算法
'''
from numpy import *

class cluster_node:
    def __init__(self, vec, left = None, right = None, distance = 0.0, id = None, count = 1):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
        self.count = count

def L2dist(v1, v2):
    return sqrt(sum((v1 - v2)**2))

def L1dist(v1, v2):
    return sum(abs(v1 - v2))

def hcluster(features, distance = L2dist):
    distances = {}
    currentclustid = -1

    clust = [cluster_node(array(features[i]), id=i) for i in range(len(features))]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                if(clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0
                    for i in range(len(clust[0].vec))]

        newCluster = cluster_node(array(mergevec), left=clust[lowestpair[0]],
                                  right=clust[lowestpair[1]], distance=closest, id=currentclustid)

        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newCluster)
    return clust[0]

def extract_clusters(clust, dist):
    clusters = {}
    if clust.distance < dist:
        return [clust]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = extract_clusters(clust.left, dist=dist)
        if clust.right != None:
            cr = extract_clusters(clust.right, dist=dist)
        return cl + cr

def get_cluster_elements(clust):
    if clust.id >= 0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_elements(clust.left)
        if clust.right != None:
            cr = get_cluster_elements(clust.right)
        return cl + cr

def printClust(clust, labels = None, n=0):
    for i in range(n):
        print('')
    if clust.id < 0:
        print('-')
    else:
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])

    if clust.left != None:
        printClust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printClust(clust.right, labels=labels, n=n+1)

def getheight(clust):
    if clust.left == None and clust.right == None:
        return 1
    return getheight(clust.left) + getheight(clust.right)

def getdepth(clust):
    if clust.left == None and clust.right == None:
        return 0
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance

