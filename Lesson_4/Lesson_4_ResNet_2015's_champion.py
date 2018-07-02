#!/usr/bin/env python
# coding: utf-8

"""
@author: FrankCui
@Time: 2018.07.01

"""

"""
ResNet设计思想:
    1. 解决问题: 深层神经网络训练时, 因在误差反传时, 梯度越来越小, 甚至消失;
        1) 更宽的模型(通道数)一般不如更深的模型效果好(深度): 
            变深, 复杂度线性增加; 
            变宽, 复杂度平方增加,易overfitting.
    2. 残差网络: 
            残差: 并联两个通道, 一个浅的小网络, 一个深层大网络, 小网络易于训练, 但模型复杂度不够, 没有拟合到的部分叫'残差', 由深层大网络拟合.
ResNet优点: 并不必GoogLeNet好, 深知GoogLeNet效率更高, ***ResNet为当前主流网络**
    1. 结构清晰, 易懂;
    2. 设计简单, channel数具有规律型, kernel值一致, 扩展性好(深度扩展仅需添加几行);
    3. 容易训练(具有jumpy跳跃通路) 
"""

from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn


class Residual(nn.Block):
    """
    Residual网络块:
        包括两个通道: path 1 和 path 2;
            1) path 1 包括: 两层卷积, 每层卷积包括kernel_size=3的卷积 + batich Norm + 'relu'激活;
            2) path 2 包括: 对input直接输出或改变channel输出(根据是否same_shape)

    """
    def __init__(self, channels, same_shape=True, **kwargs):
        """若same_shape=True, 则传入的channel值应与输入数据channel值一致"""
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            # 根据same_shape"输入与输出"图样是否同尺寸, 决定卷积步长strides
            strides = 1 if self.same_shape else 2   # v = var1 if condit else var2  --> condit 为True, 则v = var1, 否则v = var2

            # path 1 的第一层卷积: relu激活
            self.conv_1 = nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=strides)
            self.bn_1 = nn.BatchNorm(axis=1)        # 沿channel方向批量归一化
            # path 1 的第二层卷积: 未relu激活, strides始终为1, 图样尺寸不变
            self.conv_2 = nn.Conv2D(channels=channels, kernel_size=3, padding=1)
            self.bn_2 = nn.BatchNorm(axis=1)

            # path 2 : 保持图样输出尺寸与path 1 的输出尺寸一致
            if not same_shape:
                self.conv_3 = nn.Conv2D(channels=channels, kernel_size=1, strides=strides)  # path 1图样尺寸减半时, path 2 也减半

    def forward(self, X):
        """override forward method"""
        # path 1
        out_path_1 = nd.relu(self.bn_1(self.conv_1(X)))
        out_path_1 = self.bn_2(self.conv_2(out_path_1))
        # path 2
        out_path_2 = X if self.same_shape else self.conv_3(X)
        return out_path_1 + out_path_2


class ResNet(nn.Block):
    """
    ResNet类, 包括6个Block网络块:
        1. block 1: 一个卷积层;
        2. block 2: 一层MaxPool池化 + 两层Residual网络块, channel不变, 尺寸不变
        3. block 3: 两层Residual网络块, channel大一倍, 尺寸小一倍
        4. block 4: 两层Residual网络块, channel大一倍, 尺寸小一倍
        5. block 5: 两层Residual网络块, channel大一倍, 尺寸小一倍
        6. block 6: 一层AvgPool池化层 + 一层Dense全连接

    """
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.verbose = verbose
        with self.name_scope():
            # Block 1: 一层卷积Conv2D
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)

            # Block 2: 一层Max池化 + 两层Residual网络块
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )

            # Block 3: 两层Residual网络块
            b3 = nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )

            # Block 4: 两层Residual网络块
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )

            # Block 5: 两层Residual网络块
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )

            # Block 6: 一层AvgPool池化 + 一层Dense全连接
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )

            # chain all Blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, X):
        """override the forward method"""
        out = X
        for index, block in enumerate(self.net):
            out = block(out)
            if self.verbose:
                print('block %d out.shape is: %s' % (index+1, out.shape))
        return out



if __name__=='__main__':

    # 实例化Residual
    # res = Residual(3, same_shape=False)
    # res.initialize()
    #
    # X = nd.random.uniform(shape=(5, 8, 16, 16))
    # y = res(X)

    # print('y.shape is: ', y.shape)
    # print("out_path_1.shape is: ", res.out_path_1.shape)
    # print("out_path_2.shape is: ", res.out_path_2.shape)

    resnet = ResNet(10, True)
    resnet.initialize()

    X = nd.random.uniform(low=0, high=10, shape=(5, 3, 128, 128))
    y = resnet(X)
