#!/usr/bin/env python
# coding: utf-8


"""
采用gluon实现AlexNet网络：2012年 ImageNet 冠军算法
本质：
    LeNet算法加强版
"""

from mxnet import gluon
from mxnet import nd
from mxnet import image
from mxnet import autograd
from mxnet import init

import utils


"""
Section_1: 定义AlexNet模型
AlexNet分析:
    缺点:
"""
net = gluon.nn.Sequential()
with net.name_scope():
    # 第一阶段：一层卷积+一层池化
    net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    
    # 第二阶段：一层卷积+一层池化
    net.add(gluon.nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    
    # 第三阶段：三层卷积+一层池化
    net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
    net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
    net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    
    # 第四阶段：flatten扁平化 + Dense全连接 + dropout正则化
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(4096, activation='relu'))
    net.add(gluon.nn.Dropout(0.5))
    
    # 第五阶段：Dense全连接 + dropout正则化
    net.add(gluon.nn.Dense(4096, activation='relu'))
    net.add(gluon.nn.Dropout(0.5))
    
    # 第六阶段：Dense层输出
    net.add(gluon.nn.Dense(10))

"""***************************************************************************************************************
Section_2: 数据集处理
"""

# 加载数据集
def load_data_set(batch_size, re_size):
    train_data, test_data = utils.load_data_fashion_mnist(batch_size, resize=re_size)
    return train_data, test_data


# 测试: 查看data和label的shape
# train_data, test_data = load_data_set(64, 224)
# for data, label in train_data:
#     print('data.shape is: ', data.shape)
#     print('data[2] is: ', data[2].max())
#     print('label.shape is: ', label.shape)
#     break

"""************************************************************************************************************
Section_3: 开始训练：
要点:
     1. 使用Xavier初始化参数；
     2. 使用更小学习率
    
"""
# 模型初始化
ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

# 设置训练参数
learning_rate = 0.01
epochs = 1
batch_size = 64
re_size = 224

# 获取训练集和测试集
train_data, test_data = load_data_set(batch_size, re_size)

# 定义损失函数和优化器
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

# 优化迭代epochs次
for epoch in range(epochs):
    print('training start...')
    train_loss = 0.
    train_acc = 0.

    # 每次取出batch_size个数据
    for data, label in train_data:
        with autograd.record():
            output = net(data.as_in_context(ctx))   #数据集迁移只ctx设备
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc = utils.accuracy(output, label)
        
    test_acc = utils.evaluate_accuracy(test_data, net, ctx)

    print('Epoch: %d.\nTrain loss: %f.\nTrain acc: %f.\nTest acc: %f' %\
             (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
    print('training End!')
