# -*- coding: utf-8 -*-
# author: 张艳升

'''
自己实现简单的network
'''
import random
import numpy as np

class NetWork(object):
    '''
    初始化网络结构 sizes为列表包含每一层有多少神经元
    '''
    def __init__(self,sizes):
        self.num_layers = len(sizes) #网络层数
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #初始化偏向
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1],sizes[1:])] #初始化权重
    '''
    前向传播函数，往前直到输出层
    '''
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    '''
    经典往后传递要用到
    随机梯度下降算法
    training_data:训练集
    epochs:循环迭代次数
    mini_batch_size:每一次取一批数据
    eta:学习率，也就是步长
    test_data:测试集
    '''
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0,n,mini_batch_size)]
            for mini_batch in  mini_batches:
                self.update_mini_batch(mini_batch, eta) #核心
            if test_data:
                print("Epoch{0}: {1}/{2}".format(j, self.evaluate(test_data),n_test))
            else:
                print("Epoch{0} complete".format(j))

    '''
    往后传递，进行更新
    更新权重与偏向
    '''
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)#核心
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w ,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases,nabla_b)]

    '''
    向后传递更新
    得到权重与偏向的导数
    '''
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x]
        #zs储存中间变量z的list
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
         #backward pass
        delta = self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])#Error 误差
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    '''
    评估函数
    '''
    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)),y)
                        for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    '''
    真实值与预测值的差值
    '''
    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

'''
激活函数，sigmoid = 1.0/(1.0+exp(-z))
神经元输出时用sigmoid转换
'''
def sigmoid(z):
    return (1.0/(1.0+np.exp(-z)))

'''
激活函数的导数
sigmoid(z)‘ = sigmoid(z)*(1-sigmoid(z))
'''
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
