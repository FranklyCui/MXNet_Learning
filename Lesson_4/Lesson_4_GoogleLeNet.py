#!/usr/bin/env python
# coding: utf-8

"""
@author: FrankCui
Method: 在fashion minist数据集上, 比较不同的网络的结果
"""

'''
设计思想: GoogLeNet 是N个Inception的嵌套
    1. 尽可能收集信息: 3条不同尺寸kernel + 1条MaxPooling
        1)  kernel_size 越大, 能够catch的信息越多, 
        2) 不清楚该选用什么尺寸的kernel, 故把3种不同尺寸的kernel开在一起, 并行处理;
        3)  第4条Path开一条MaxPooling, 代替串联卷积Conv和池化MaxPooling
    2. 降维: 每条path通路内添加kernel_size为1的卷积层Conv2D, 以减小多条path通道concat后的channel数, 减小计算复杂度.
               
         ** 计算复杂度和input及output的channel数均为平方关系 **
'''

import sys
sys.path.append('..')
import utils

from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
from mxnet import init


class Inception(nn.Block):
    """    继承Block类, 自定义一个Inception对象    """
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        with self.name_scope():
            """ 经卷积计算后, 图样尺寸: N = (W − F + 2P )/S+1 """
            # path 1: 1个kernel为1的卷积
            self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')

            # path 2: kernel为1的卷积 + kernel为3的卷积
            self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
            self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1, activation='relu')

            # path 3: kernel为1的卷积 + kernel为5的卷积
            self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
            self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2, activation='relu')

            # path 4: Max池化 + kernel为1的卷积
            self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
            self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, X):
        """override forward方法"""
        # path 1
        p1 = self.p1_conv_1(X)
        # path 2
        p2 = self.p2_conv_3(self.p2_conv_1(X))   # 第一层kernel=1的channel数通常小于第二层kernel=3的channel数, 以减小后者计算量
        # path 3
        p3 = self.p3_conv_5(self.p3_conv_1(X))
        # paht 4
        p4 = self.p4_conv_1(self.p4_pool_3(X))

        # 每个path的输出为: batch x channel x height x weight
        # 按照channel进行拼接, 故: dim = 1
        return nd.concat(p1, p2, p3, p4, dim=1)

class GoogLeNet(nn.Block):
    """
    GoogLeNet类:
        1) 6个Block, 前5个为Conv或Inception, 每个Block之间, 采用strides=2, kernel_size=3 的Max池化层缩减图样尺寸;
        2) 第一个Block: 1个channel=64, kernel_size=7 的卷积层
        3) 第二个Block: 2个卷积层, kernel_size=1的卷积层 + kernel_size=3的卷积层;
        4) 第三个Block: 2个串联的Inception;
        5) 第四个Block: 5个串联的Inception;
        6) 第五个Block: 2个串联的Inception;
        7) 最后: 1层Dense全连接.

    """
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1: 一层卷积层 + 一层MaxPool池化层
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3, activation='relu'),   # 为更多的提取原始图样信息, 第一层卷积的kernel比较大
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # Block 2: 两层卷积层, kernel_size=1的卷积层 + kernel_size=3的卷积层 + MaxPool池化层
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(channels=64, kernel_size=1),            # 未添加activation激活函数
                nn.Conv2D(channels=192, kernel_size=3, padding=1), # 未添加activation激活函数
                nn.MaxPool2D(pool_size=3, strides=2, padding=1)   # 添加padding
            )

            # Block 3: 2个串联的Inception + MaxPool池化层
            b3 = nn.Sequential()
            b3.add(
                Inception(64, 96, 128, 16, 32, 32),
                Inception(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # Block4: 5个串联的Inception + MaxPool池化层
            b4 = nn.Sequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # Block 5: 2个串联的Inception + AvgPool池化层
            b5 = nn.Sequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),

                nn.AvgPool2D(pool_size=2)    # 采用AvgPool平均池化, 将图样尺寸batch x channel x height x weight 转化为batch x channel x 1 x 1
            )
            # Block 6: 一层Dense全连接
            b6 = nn.Sequential()
            b6.add(
                nn.Flatten(),
                nn.Dense(num_classes)
            )

            # chain Block together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, X):
        """override the forward method"""
        out = X
        for index, block in enumerate(self.net):
            out = block(out)
            if self.verbose:
                print("%d Block's out.shape is: %s" % (index, out.shape))
        return out


if __name__ == '__main__':

    # # 实例化Inception
    # incp = Inception(64, 96, 128, 16, 32, 32)
    # incp.initialize()
    #
    # X = nd.random.uniform(shape=(32, 3, 64, 64))
    # print(incp(X).shape)

    # # 实例化 GoogleNet
    # net = GoogLeNet(10, verbose=True)
    # net.initialize()
    # X = nd.random.uniform(shape=(4, 3, 96, 96))
    # out = net(X)
    # print('out of net is: ', out)

    """
    训练模型
    """
    train_data, test_data = utils.load_data_fashion_mnist(batch_size=4, resize=96)

    ctx = utils.try_gpu()
    net = GoogLeNet(10)
    net.initialize(ctx=ctx, init=init.Xavier())

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    utils.train(train_data, test_data, net=net, loss=loss, trainer=trainer, ctx=ctx, num_epochs=1)


