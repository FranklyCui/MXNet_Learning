#!/usr/bin/env python
# coding: utf-8

"""
nin (network in network) 思想:
    1. net中嵌套net;
    2. "一卷到底", 采用卷积Conv2D + 平均池化AvgPooling 代替全连接Dense.
优点:
    1. 避免全链接Dense的大计算资源消耗, 特别是存储资源消耗(显存); # AlexNet的750M, 近500M为全连接Dense层;
    2. 减小因全连接Dense层导致的过拟合;

"""


import sys
sys.path.append('..')

import utils

from mxnet import gluon
from mxnet import init

# 思想: 每层卷积后, 跟两个kernel为1的卷积层Conv2D, 代替全连接Dense
def mlp_conv(channels, kernel_size, padding, strides=1, max_pooling=True):
    """代码块: 包括1个正常卷积层 + 2个kernel为1的卷积层(充当Dense全连接层)角色"""
    out = gluon.nn.Sequential()
    with out.name_scope():
        out.add(gluon.nn.Conv2D(channels=channels, kernel_size=kernel_size, \
                                strides=strides, padding=padding, activation='relu'))
        # kernel置为1×1, 用以代替Dense全链接层: stride=1, padding=1
        out.add(gluon.nn.Conv2D(channels=channels, kernel_size=1, \
                                strides=1, padding=0, activation='relu'))
        out.add(gluon.nn.Conv2D(channels=channels, kernel_size=1, \
                                strides=1, padding=0, activation='relu'))
        if max_pooling:
            out.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    return out


# 思想: 多个mlp_conv代码块后, 采用通道数与目标类数相同的AverPooling代替全连接层Dense
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(
        mlp_conv(channels=96, kernel_size=11, padding=0, strides=4),
        mlp_conv(channels=256, kernel_size=5, padding=2),
        mlp_conv(channels=384, kernel_size=3, padding=1),

        # 正则化
        gluon.nn.Dropout(rate=0.5),

        # 目标类数: 10
        mlp_conv(channels=10, kernel_size=3, padding=1, max_pooling=False),
        # 进行average池化, 代替全连接层Dense
        # 输入尺寸: batch_size x 10 x 5 x 5 ; 输出为: batch_size x 10 x 1 x 1
        gluon.nn.AvgPool2D(pool_size=5),   # pool_size为输入图片的尺寸值

        # 利用Flatten转成二维矩阵: batch_size x 10
        gluon.nn.Flatten()
    )

# 获取数据
train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=224)
# 获取设备context
ctx = utils.try_gpu()

# 初始化模型
net.initialize(ctx=ctx, init=init.Xavier())

# 实例化损失函数
loss = gluon.loss.SoftmaxCrossEntropyLoss()
# 实例化优化器
learning_rate = 0.1
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)


