#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet import image
from mxnet import nd

# ir = open('/home/frank/gluon-tutorials-zh/img/cat1.jpg', 'rb')
with open('/home/frank/gluon-tutorials-zh/img/cat1.jpg', 'rb') as ir:
    img = image.imdecode(ir.read())
plt.imshow(img.asnumpy())

# print('img is: ', img.shape, img[0][0])
# print('img.asnumpy is: ', img.asnumpy.shape, img.asnumpy[0])

def apply(img, aug, n=3):
    """
    辅助函数:
        输入img图片和aug增强方法, 绘制出增强后的n*n个结果
    """
    _, figs = plt.subplots(n, n, figsize=(8, 8))        # plt.subplots返回一个figure对象和一个axes对象列表

    for row in range(n):
        for column in range(n):
            # 将img的类型由int8转为float32格式
            x = img.astype('float32')
            # 保证输入是合法值
            y = aug(x).clip(0, 254)

            # 显示浮点图片时, imshow()要求 元素需在[0, 1]之间
            y_numpy = y.asnumpy() / 255.0
            # row行column列的axes子图, 绘图并设置坐标轴不可见
            figs[row][column].imshow(y_numpy)
            figs[row][column].axes.get_xaxis().set_visible(False)
            figs[row][column].axes.get_yaxis().set_visible(False)

def aug_method():
    # 增强方法1: 图片变形
    # 1) 水平翻转: 最常用!! 对卷积神经网络而言, 左右互换(水平翻转)差异很大
    aug = image.HorizontalFlipAug(0.5)
    apply(img, aug)
    plt.gcf().suptitle('HorizontialFlipAug')
    plt.show()

    # 2) 随机裁剪: 卷积神经网络对位置敏感, 虽然Pool池化可以减小位置敏感, 但仍不不够
    # 2.1) 裁剪指定(width x height) 的图片
    aug = image.RandomCropAug((200, 200))
    apply(img, aug)
    plt.show()
    # 2.2) 裁剪指定'长宽比'范围大小且不小于最小值的随机图片, 并resize为指定(width x height)大小
    aug = image.RandomSizedCropAug(size=(100, 100), min_area=.1, ratio=(0.5, 2.0))
    apply(img, aug)
    plt.show()

    # 增强方法2: 颜色变化
    # 1) 亮度变化:
    aug = image.BrightnessJitterAug(brightness=0.5)     # brightness : The brightness jitter ratio range, [0, 1]
    apply(img, aug)
    plt.show()

    # 2) 色调变化:
    aug =image.HueJitterAug(hue=0.5)
    apply(img, aug)
    plt.show()

"""*********************************************************************************************************************
    对CIFAR10数据集进行图片增强.
"""

def apply_aug_list(img, augs):
    """对输入图片, 按照augs列表中aug方法逐个增强"""
    for aug in augs:
        img = aug(img)
    return img

# 训练集增强方法
train_augs = [
    image.HorizontalFlipAug(0.5),
    image.RandomCropAug(size=(28, 28))
]

# 测试集增强方法
test_augs = [
    image.CenterCropAug(size=(28, 28))
]

def get_transform(augs):
    """  """
    def transform(data, label):
        """  """
        data = data.astype('float32')
        if augs is not None:
            data = apply_aug_list(data, augs)
        data = nd.transpose(data, axes=(2, 0, 1)) / 255
        return data, label.astype('float32')
    return transform


def get_data(batch_size, train_augs, test_augs=None):
    """ 获取训练集和测试集, 并对数据集进行增强 """
    # 获取"增强后的"数据集
    cifar10_train = gluon.data.vision.CIFAR10(
        train=True, transform=get_transform(train_augs))
    cifar10_test = gluon.data.vision.CIFAR10(
        train=False, transform=get_transform(test_augs))

    # 生成batch大小的数据块迭代器
    train_data = gluon.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
    return train_data, test_data

def draw_batch_image():
    """ 绘制batch张图看下 """
    train_data, _ = get_data(batch_size=36, train_augs=train_augs)

    # 获取一个batch的图像集
    def get_batch_imgs():
        for imgs, _ in train_data:
            print('imgs.shape is: ', imgs.shape)
            return imgs

    # 绘制一长6x6的画布Figure, 并获取子图axes列表
    _, axes = plt.subplots(nrows=6, ncols=6, figsize=(25, 25))
    # 获取batch=36张图片
    imgs = get_batch_imgs()
    # 遍历每个axe子图, 绘制图片
    for row in range(6):
        for col in range(6):
            # 将mxnet的 channel x width x height 转为 width x height x channnel
            img = nd.transpose(imgs[row*6 + col], axes=(1, 2, 0))       # vedio里面源码为: row * 3, 错误, 应为 row * 6
            # 坐标轴设置为不显示
            axes[row][col].imshow(img.asnumpy())
            axes[row][col].axes.get_xaxis().set_visible(False)
            axes[row][col].axes.get_yaxis().set_visible(False)
    plt.show()


def show_image(images):
    """ 绘制传入的图像集 batch_size x width x height x channnel , 并可视化显示"""
    num_img = images.shape[0]
    figsize = images.shape[1:3]
    _, axes = plt.subplots(nrows=1, ncols=num_img, figsize=figsize)

    for col in range(num_img):
        # image转化为numpy格式
        img = images[col].asnumpy()
        axes[col].imshow(img)
        axes[col].axes.get_xaxis().set_visible(False)
        axes[col].axes.get_yaxis().set_visible(False)
    plt.show()



if __name__=="__main__":
    import numpy as np
    cifar10 = gluon.data.vision.CIFAR10()
    data, label = cifar10[0: 9]
    print('cifar10[0].shape is: ', np.shape(cifar10[0]))
    # print('data is: ', data)
    print('data.shape is: ', data.shape)
    print('label is: ', label)
    show_image(data)





