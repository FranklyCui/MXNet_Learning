#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet import image

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

"""*****************************************************************************************

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


