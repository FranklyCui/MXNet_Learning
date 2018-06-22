#!usr/bin/env python
# coding: utf-8

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import numpy as np

import sys
import utils

# 构造数据集迭代器
sys.path.append('..')
def load_data_set(batch_size):
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    return train_data, test_data


# 尝试使用GPU运算：try/except，
# 思路：（1）先尝试用GPU构造数组，(2) 若成功，则令ctx=mx.gpu(),（3）若无GPU，则报错，令ctx = mx.cpu()
def try_ctx():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

'''
Section_1: 手动利用NDarray函数直接实现LetNet卷积网络
'''

'''
利用
******
卷积网络params处理：

**1）李沐讲解：
    data.shape   ： batch * channel * height * width
    weight.shape ： input_filter * output_filter * height *width

*)2）个人理解：
    num_filter: 便是channel，weight.shape内，
                shape[0]为本层神经元数，该数也构成下层网络的输入channel（下层的shape[1])；
                shape[1]为上层神经元树，该数也即上层网络的输出channel（上层的shape[0]）；
                shape[2:3]为图片尺寸
*)3）引自Document API：
    data:   (batch_size, channel, height, width)
    weight: (num_filter(本输出层的神经元个数), channel, kernel[0], kernel[1])
    bias:   (num_filter,)
    out:    (batch_size, num_filter, out_height, out_width).

******
'''
def initialize_params():
    # 初始化参数设置
    weight_scale = 0.01  # weight标准差
    num_output = 10  
    num_fc = 128  # 特征数

    # 两层卷积：权重、截距，初始值
    W1 = nd.random_normal(shape=(20, 1, 5, 5), scale = weight_scale, ctx = ctx)
    b1 = nd.zeros(W1.shape[0])
    W2 = nd.random_normal(shape=(50,20,3,3), scale=weight_scale, ctx=ctx)
    b2 = nd.zeros(W2.shape[0])

    # 两层全连接层：权重、截距，初始值
    W3 = nd.random_normal(shape=(1250,128), scale=weight_scale, ctx=ctx)
    b3 = nd.zeros(shape=W3.shape[1], ctx=ctx)
    W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale,ctx = ctx)
    b4 = nd.zeros(shape=W4.shape[1], ctx=ctx)

    # 创建待优化params，并开辟梯度空间
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    for param in params:
        param.attach_grad()
    return params
    
# 定义LetNet模型
def LetNet_Direct(X, verbose=False):
    # 初始化参数
    params = initialize_params()
    W1, b1, W2, b2, W3, b3, W4, b4 = params
    # 1. 将数据集copy到GPU
    X = X.as_in_context(ctx)
    
    # 2. 定义卷积层
    # 2.1 卷积层一
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1,  kernel=W1.shape[2:],num_filter=W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2,2),stride=(2,2))
    # 2.2 卷积层二
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2,kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2,2), stride=(2,2))  #Plooing的Kernel值，决定输出图像大小
    # Flatten成2-dimens矩阵，以作为Dense层输入
    h2 = nd.flatten(h2)    

    # 3. 定义全连接层
    # 3.1 第一层全连接：激活函数非线性
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    # 3.2 第二层全连接
    h4 = nd.dot(h3_linear, W4) + b4
    
    # 是否显示详细信息：各层代码块的输出shape
    if verbose:
        print('1st conv block: ', h1.shape)
        print('2st conv block: ', h2.shape)
        print('3st dense: ', h3.shape)
        print('4st dense: ', h4.shape)
    return h4

'''
Section_2: 采用GLuon实现LetNet卷积网络
优点：
    无需关注输入层的channel大小，只需定义（1）本层channel(num_filter)大小、（2）kernel大小，以及(3)激活函数和(4)步长
'''
def LetNet_Gluon():
    print('start run...')
    
    batch_size = 256
    ctx = try_ctx()
    
    # Step_1: 定义模型(容器)
    net = gluon.nn.Sequential()
    # Step_2: 在模型(容器)域名内，添加两层卷积和池化层、两层Dense层。
    with net.name_scope():
        # 两层卷积Conv2D网络
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5,activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=3,activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # 卷积网络输出结果扁平化处理
        net.add(gluon.nn.Flatten())
        # 两层Dense网络
        net.add(gluon.nn.Dense(128, activation='relu'))
        net.add(gluon.nn.Dense(10))

    # Step_3: 初始化模型
    net.initialize(ctx=ctx)
    print('initialize weight on', ctx)

    # Step_4: 构造损失函数：交叉熵
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # Step_5: 构造优化器：SGD
    trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate': 0.5})

    # 优化迭代epochs次
    epochs = 5
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        
        # 每次取出batch_size(256)个数据，遍历数据集        
        train_data, test_data = load_data_set(batch_size)
        for data, label in train_data:
            # 将数据迁移置GPU（ctx设备）
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # 记录梯度：计算输出和损失函数时
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            # 反向传播
            loss.backward()
            # 优化器：更新参数
            trainer.step(batch_size)
            
            # 统计整个数据集的loss和acc：对数据集的每个batch_size大小数据块的平均loss和平均acc累加（应再/len(data)以求平均值）
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)        
        test_acc = utils.evaluate_accuracy(test_data, net, ctx)        
        print('Epoch %d. Train loss is: %f;\nTrain acc is: %f;\nTest acc is: %f;\n' % \
                (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
   
    print("run over!")
    return net

# 测试函数
def test_func():
    LetNet_Gluon()
    for data, _ in train_data:
        #LetNet_Direct(data, verbose=True)
        break
