from mxnet import gluon
from mxnet import nd

"""
Section_1: VGG模型实现
思想: 
    1) 通过比较简单的模块化定义,比较简单的构造网络;  (AlexNet比较飘逸的参数选择法,不适合推广使用) 
    2) 经过多层卷积,图样尺寸不断变小(每层input),每层网络神经元数不断增加(每层channels) ---> **通用的设计思想**

"""

# 经过一个block,图样尺寸缩小一倍
def vgg_block(num_convs, channels):
    """定义num_convs个卷积层 + 1个MaxPool池化层的网络块"""
    net = gluon.nn.Sequential()

    with net.name_scope():
        for _ in range(num_convs):
            # 图样尺寸公式:  W--> (W + 2P- F ) / S + 1;
            net.add(gluon.nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))   #图样尺寸不变
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))    # 图样尺寸缩小1倍
    return net


# 实例化并初始化
blk = vgg_block(2, 128)
blk.initialize()

# x = nd.random.uniform(shape=(2, 3, 16, 16))
# print('blk is: \n', blk)
# print('type of blk is: ', type(blk))
# y = blk(x)
# print('x.shape is: ', x.shape)
# print('y.shape is: ', y.shape)
# """(16 - 3 + 2)/1 + 1 = 16; 16; (16 - 2)/2 + 1 = 8"""

# 根据传入参数,stack多个Block
def vgg_stack(architecture):
    """传入给定参数集(列表),调用vgg_block构造vgg_stack"""
    net = gluon.nn.Sequential()
    with net.name_scope():
        # 循环调用vgg_block函数添加网络块
        for (num_convs, channels) in architecture:
            net.add(vgg_block(num_convs, channels))
    return net


"""
Section_2: 定义VGG11: 8个卷积层 + ３个 AlexNet一样的２个全连接层　+ 输出层
"""
# 模型参数
num_output = 10
arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

Model = gluon.nn.Sequential()
with Model.name_scope():
    # 调用VGG卷积块
    Model.add(vgg_stack(arch))
    Model.add(gluon.nn.Flatten())

    # 3个AlexNet全连接层 + 输出层
    Model.add(gluon.nn.Dense(4096, activation='relu'))
    Model.add(gluon.nn.Dropout(0.5))
    Model.add(gluon.nn.Dense(4096, activation='relu'))
    Model.add(gluon.nn.Dropout(0.5))
    Model.add(gluon.nn.Dense(num_output))

print('Model is: ', Model)
Model.initialize()
print('Model is: ', Model)
