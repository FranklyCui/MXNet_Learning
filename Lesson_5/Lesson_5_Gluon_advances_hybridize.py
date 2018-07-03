#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@author: FrankCui
"""

"""
要点: 
    使用方法:  确保网络的所有层均继承自HybridBlock.
        1. 先使用命令式编程开发和调试, 
        2. 而后, 调用net.hybridize()方法, 切换为符号式编程, 生成计算图;
        3. 最后, 使用net.export()方法保存模型.


Gluon的高级特性: 小的网络不会觉得, 但复杂网络非常重要, 可加速.
    1. 符号式编程: 先构建计算图, 而后编译, 再运行;
            优点: 省内存, 性能更好, 移植性好;
            缺点: debug困难, 与主流语法交互困难.
"""

from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn

"""*************************************************************************************
# Gloun支持: 先采用命令式编程开发和调试, 到产品性能和部署阶段, 切换到符号式编程;
"""

def get_net():
    """ 返回一个初始化后的Hybrid网络 """
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(3)
        )
    net.initialize()
    return net

# 准备数据, 获取初始化的网络
x = nd.random.normal(shape=(5, 3, 8, 8))
net = get_net()

# 命令式编程
y = net(x)
print('y is: ', y)

# 符号式编程
net = get_net()
net.hybridize()         # 将命令式编程转化为符号式编程
y = net(x)
print('y is: ', y)

"""*************************************************************************************
# 比较hybridize()前后的时间差, 即符号式和命令式编程的性能
"""
from time import time

def bench(net, x):
    """ 返回计算1000次net(x)耗时 """
    start = time()
    for i in range(1000):
        y = net(x)
    # 等待所有计算完成
    nd.waitall()
    return time() - start

net = get_net()
cost_time = bench(net, x)
print('imperative coding cost time is: %0.4f sec' % cost_time)

net.hybridize()
cost_time = bench(net, x)
print('symbol coding is: %0.4f sec' % cost_time)

"""*************************************************************************************
# 调用hybridize()前后的区别:
    1. 调用前, 传入数据应为NDArray类型, net(x)返回NDArray类型;
    2. 调用后, 已生成计算图, 可传入Symbol类型变量, 返回Symbol类型
    
    缺点:
        1. 损失灵活性
"""
from mxnet import sym

# x = sym.var('data')
# y = net(x)
# print(y.tojson())

class HybridNet(nn.HybridBlock):
    """ 自定义一个Hybrid网络 """
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden = nn.Dense(10)
            self.out = nn.Dense(2)

    def hybrid_forward(self, F, x):
        """ override hybrid_forward method """
        print('F is: ', F)
        x = F.relu(self.hidden(x))
        print('self.hidden out is: ', x)
        return self.out(x)


x = nd.random.normal(shape=(2, 3, 8, 8))
print("-" * 100)
net = HybridNet()
net.initialize()
# 命令式编程
y = net(x)
print("y is: ", y)

# 符号式编程
print("-" * 100)
net.hybridize()
y = net(x)
print("y is: ", y)

# 再次运行: 程序不再重复调用编译, 直接调用计算图, 故编译过程的print不再输出
y = net(x)
print('-' * 100)
print('y is: ', y)

