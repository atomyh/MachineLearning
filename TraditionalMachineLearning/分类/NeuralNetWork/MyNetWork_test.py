# -*- coding: utf-8 -*-
# author: 张艳升

import mnist_loader
import MyNetWork

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
# print("training data")
# print(type(training_data))
# print(len(training_data))
# print(training_data[0][0].shape)
# print(training_data[0][1].shape)
validation_data = list(validation_data)
# print("validation_data")
# print(type(validation_data))
# print(len(validation_data))
# print(validation_data[0][0].shape)
# print(validation_data[0][1].shape)
test_data = list(test_data)

net = MyNetWork.NetWork([784,30,10])
net.SGD(training_data,30,10,3.0,test_data=test_data)



