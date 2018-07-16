#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

"""
@atuthor: FrankCui
"""

"""
微调(Fine Tuning):
    思想:
        当面对一个较小数据集, 而担心它可能不够训练出很好的模型时, 可寻找跟目标数据集类似的大数据集来训练模型, 
        而后, 对该模型依据目标数据集微调.
    原理:
        若两个任务的数据具有很多共通性, 如需了解如何识别文理, 形状, 边等, 这些特征通常在靠近数据的层得到处理.
    方法:
        1. 在源数据集(大数据集)S上训练一个神经网络模型;
        2. 去掉模型输出层, 改为目标数据的输出层;
        3. 将输出层权重随机初始化, 其他层沿用两模型参数不变;
        4. 开始在目标数据集训练.

"""

from mxnet import gluon
from mxnet import nd
from mxnet import image
from mxnet import init
from mxnet.gluon.model_zoo import vision as models

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import utils

import zipfile

data_dir = '../data/'
base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
fname = gluon.utils.download(
    base_url+'gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')

with zipfile.ZipFile(fname, 'r') as f:
    f.extractall(data_dir)

#
# # 下载hotdog数据集
# data_dir = '../data/'
# base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
# file_name = gluon.utils.download(base_url+'gluon/dataset/hotdog.zip',
#                              path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
#
# # 解压图片数据集
# with zipfile.ZipFile(file_name, 'r') as f:
#     f.extractall(data_dir)

# 绘制图片
def show_images(imgs, nrows, ncols, figsize=None):
    if not figsize:
        figsize = (ncols, nrows)
    # 创建画图和多个子图
    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for row in range(nrows):
        for col in range(ncols):
            # 绘制图片
            axs[row][col].imshow(imgs[row*ncols+col].asnumpy())
            # 坐标轴不显示
            axs[row][col].axes.get_xaxis().set_visible(False)
            axs[row][col].axes.get_yaxis().set_visible(False)
    plt.show()


# 图片增强方法
train_augs = [image.HorizontalFlipAug(0.5), image.RandomCropAug((224, 224))]
test_augs = [image.CenterCropAug((224, 224))]
# 变换函数
def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2, 0, 1))
    return data, nd.array([label]).asscalar().astype('float32')


# 对图片进行增强, 并按照指定path路径存储
train_imgs = gluon.data.vision.ImageFolderDataset(data_dir+'/hotdog/train',
                                                  transform = lambda X, y: transform(X, y, train_augs))
test_imgs = gluon.data.vision.ImageFolderDataset(data_dir+'/hotdog/test',
                                                  transform = lambda X, y: transform(X, y, test_augs))
# 采用ResNet18来训练
"""先获取改良过的ResNet, 再fine-tuning"""
# 预训练模型包括2块: 1)features: 从输入开始的大部分层, 2) classifier: 最后一层全连接层
pretrained_net = models.resnet18_v2(pretrained=True)
print('-' * 50)
print('pretrained_net.classifier is: ', pretrained_net.classifier)


# 获取ResNet, feature沿用与预训练模型, output部分随机初始化
finetune_net = models.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())


# 开始训练
def train(net, ctx, batch_size=64, epochs=10, learning_rate=0.01, wd=0.001):
    # 获取数据迭代器
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, batch_size)

    # 确保net初始化在ctx上
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    # 损失函数
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # 优化器
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learing_rate': learning_rate, 'wd':wd})

    # 开始训练
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs)

if __name__=='__main__':

    # 取出一个batch_size图像, 并显示
    data = gluon.data.DataLoader(train_imgs, 32, shuffle=True)
    for X, _ in data:
        X = X.transpose((0, 2, 3, 1)).clip(0, 255) / 255
        show_images(X, 4, 8)
        break

    ctx = utils.try_all_gpus()
    train(finetune_net, ctx)
