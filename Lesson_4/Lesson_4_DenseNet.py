#!/usr/bin/env python
# coding: utf-8

"""
DenseNet设计思想:
    仅在ResNet上更改了一处, 即:
        将Residual块中path 1 和 path 2 两个通道最后的" + ", 改为concat(借鉴GoogLeNet的inception中多通道后concat思想)
"""

from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn


def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels=channels, kernel_size=3, padding=1)
    )
    return out


class DenseBlock(nn.Block):
    """
    DenseBlock: 将Residual块两通道的" + " 改为 "concat"
    影响:
        1. 每经过一层DenseBlock的卷积, 输出的channel都增加卷积的channel数个,
                                如: out_channel = num_convs * channnel_convs + channel_X
    """
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.net.name_scope():
            for i in range(layers):
                self.net.add(conv_block(growth_rate))     # growth_rate即为block每层卷积的channel值, 意指每次channel的增加值.

    def forward(self, X):
        """override the forward method"""
        for layer in self.net:
            out = layer(X)
            X = nd.concat(X, out, dim=1)
        return X

    # # 每个Block一层卷积通道和一个旁路通道
    # def __init__(self, channels):
    #     super(DenseBlock, self).__init__(**kwargs)
    #     self.net = nn.Sequential()
    #     with self.net.name_scope():
    #         self.net.add(
    #             nn.BatchNorm(),
    #             nn.Activation('relu'),
    #             nn.Conv2D(channels=channels, kernel_size=3, padding=1)
    #         )
    #
    # def forward(self, X):
    #     out = self.net(X)
    #     return nd.concat(X, out)


def transition_block(channels):
    """
    功能:
        1. 用于将输入channnel降维到指定channel值;
        2. 用于将输入图像尺寸减半.
    :param channels:
    :return:
    """
    out = nn.Sequential()
    # ResNet v.2 版本思想: 先BatchNorm, 再relu, 最后Conv2D
    with out.name_scope():
        out.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2)
        )
    return out


# DenseNet: 若干个DenseBlock和transition_block的交互
init_channels = 64
growth_rate = 32
block_layer = [6, 12, 24, 16]
num_class = 10

def dense_net():
    """
    包括3个Block:
        Block 1: 卷积 + 最大池化;
        Block 2: DenseBlock, 一个DenseBlock + 一个transition_block
    """
    net = nn.Sequential()
    with net.name_scope():
        # Block 1: 卷积 + 最大池化
        net.add(
            nn.Conv2D(channels=init_channels, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )

        # Block 2: dense blocks(稠密块 + 过渡块)
        channels = init_channels
        for index, layers in enumerate(block_layer):
            net.add(DenseBlock(layers=layers, growth_rate=growth_rate))
            # 获取当前channel值
            channels += layers * growth_rate
            # channel值减半
            if index != len(block_layer) -1:
                net.add(transition_block(channels=channels//2))

        # Block 3:
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_class)
        )







if __name__=="__main__":

    # dblk = DenseBlock(2, 10)
    # dblk.initialize()
    #
    # x = nd.random.uniform(shape=(2, 3, 16, 16))
    # y = dblk(x)
    # print('y.shape is: ', y.shape)
    pass


