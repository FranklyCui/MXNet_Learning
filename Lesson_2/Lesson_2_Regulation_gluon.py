#!/usr/bin/env python
# coding: utf-8

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

import utils


# 生成数据集
def get_data():
    # 设置训练参数
    num_train = 20
    num_test = 100
    num_inputs = 200  # 特征个数
    true_w = nd.ones((num_inputs, 1)) * 0.01
    true_b = 0.05

    # 生成数据集
    X = nd.random_normal(shape=(num_train+num_test, num_inputs))
    y = nd.dot(X, true_w) + true_b + nd.random_normal(shape=(num_train+num_test,1))
    # 切分数据集为：训练集、测试集
    X_train, X_test = X[:num_train, :], X[num_train:,:]
    y_train, y_test = y[:num_train], y[num_train:]
    #initial_para_list = [batch_size, num_]
    return X_train, y_train, X_test, y_test

# **数据集迭代生成器：模型训练时，需要不断读取数据块，故定义一个每次函数，每次返回batch_size个随机的样本和对应的标签**
def data_iter(data, label, batch_size):
    # 创建打乱的索引列表
    idx = list(range(len(data)))
    random.shuffle(idx)
    # 以batch_size为步长，生成截取索引列表
    for i in range(0, len(data), batch_size):
        index = nd.array(idx[i:min(i+batch_size, len(data))])
        yield data.take(index), label.take(index)

'''
Section_1: Scratch版本，直接实现L2正则化
Scratch：If you do something from scratch, you do it without making use of anything that has been done before.（From Collins)
'''

#
# class ScratchRegulation(object):
#     # 初始化模型参数，创建梯度
#     def get_params():
#         w = nd.random_normal(shape=(num_inputs, 1)) * 0.1
#         b = nd.zeros(shape=(1,))
#         for param in (w, b):
#             param.attach_grad()
#         return (w, b)
#
#     # 创建模型
#     def net(X, w, b):
#         return nd.dot(X, w) + b
#
#     # 定义损失函数: L2正则化
#     def square_loss(yhat, y, C_reg):
#         return (yhat - y.reshape(yhat.shape)) ** 2 + C_reg * ((w ** 2).sum() + b ** 2)
#
#     # 定义SGD函数
#     def SGD(params, lr):
#         for param in params:
#             param[:] = param - lr * param.grad
#
#     # 创建test函数
#     def test(params, X, y, C_reg):
#         loss = square_loss(net(X, *params), y, C_reg).mean().asscalar()
#         return loss
#
#     # 创建训练函数
#     def train(C_reg):
#         # 初始化参数
#         epochs = 10
#         learning_rate = 0.002
#         batch_size = 1
#         params = get_params()
#
#         # 获取数据
#         X_train, y_train, X_test, y_test = get_data()
#
#         # 记录每次迭代的loss值
#         train_loss = []
#         test_loss = []
#
#         # 迭代epoch次
#         for epoch in range(epochs):
#             # 遍历数据集
#             for data, label in data_iter(X_train, y_train, batch_size):
#                 # step_1: 前向传播计算output和loss时，记录局部梯度值
#                 with autograd.record():
#                     output = net(data, *params)
#                     loss = square_loss(output, label, C_reg)
#                 # step_2：反向传播，计算output相对于input的梯度值
#                 loss.backward()
#                 # step_3: 根据梯度值和学习率，对参数进行一次迭代更新
#                 SGD(params, learning_rate)                      # param.grad是batch_size个样本点的均值梯度，还是random梯度？
#             train_loss.append(test(params, X_train, y_train, C_reg))
#             test_loss.append(test(params, X_test, y_test, C_reg))
#
#         # 绘制loss图
#         plt.plot(train_loss)
#         plt.plot(test_loss)
#         plt.legend(['train', 'test'])
#         plt.show()
#
#         print('learned w[:10]: ', params[0][:10], 'learned b: ', params[1])
#
#         return params
#

class GluonRelation(object):
    # 构造损失函数： L2Loss即平方损失（gluon版本正则化不在损失函数中添加，而在梯度下降中）
    square_loss = gluon.loss.L2Loss() 
    
    def __init__(self, batch_size, weight_decay):
        super().__init__()
        self.batch_size = batch_size
        self.weight_decay = weight_decay
    
    # 创建模型
    def net(self):
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(1))
        net.initialize()
        print('net is: ', net)
        return net

    # 创建test函数
    def test(self, X, y):
        net = self.net()
        loss = self.square_loss(net(X), y).mean().asscalar()
        return loss

    # 创建训练函数
    def train(self):
        # 初始化参数
        epochs = 20
        learning_rate = 0.002
        
        # 获取数据
        X_train, y_train, X_test, y_test = get_data()

        # 定义模型
        net = self.net()
         
        # 定义优化器对象: gluon 版本，添加权值weight_decay
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': self.weight_decay})
        
        # 记录每次迭代的loss值
        train_loss = []
        test_loss = []
        
        # 迭代epoch次
        for epoch in range(epochs):
            # 遍历数据集
            for data, label in data_iter(X_train, y_train, self.batch_size):
                with autograd.record():
                    output = net(data)
                    loss = self.square_loss(output, label)
                loss.backward()
                # 梯度下降
                trainer.step(self.batch_size)    # gluon 版本: 需先建立trainer()对象
                # SGD(params, learning_rate)  #scrathc 版本： 需先定义SGD（）函数
            # 记录每次迭代的loss值    
            train_loss.append(self.test(X_train, y_train))
            test_loss.append(self.test(X_test, y_test))
        # 绘制loss图
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
           
        #print('net[0].weight_decay is: ', net.weight_decay)
        print('learned weight', net[0].weight.data(), 'learned bias', net[0].bias.data())
        return self.net


# if __name__ == 'main':
#     gluon_class = GluonRelation()
#     gluon_class.train(0.1)#


gluon_class = GluonRelation(5, 0.1)

print(gluon_class)
gluon_class.train()
