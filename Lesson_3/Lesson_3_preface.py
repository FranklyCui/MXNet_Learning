#!/user/bin/env python
# coding: utf-8


import mxnet as mx
from mxnet import nd
from mxnet import gluon


'''*******************************************************************************************************************
Section_1: 自定义net对象、层
要点：
    1. 继承gluon.nn.Block来自定义net对象，需重写：__init__（)和forward()方法，forwar()方法会自动绑定net(X)；
    2. 自定义
'''


# 1. 利用gluon.nn.Block自定义net对象
class MLPDemon(gluon.nn.Block):
    """自定义net对象，需重写__init__和forward()两个方法"""
    def __init__(self, **kwargs):
        super(MLPDemon, self).__init__(**kwargs)
        with self.name_scope():         # 采用name_scope()方法为对象添加prefix前缀
            self.dense0 = gluon.nn.Dense(256)
            self.dense1 = gluon.nn.Dense(10)
        
    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))
# 实例化
Model = MLPDemon()
Model.initialize()


# 自定义一个简单特定功能的层：功能：输入-均值；缺点：不具有优化参数
# 自己的理解：只需要override构造方法和forward方法即可
# 实例化后：layer(x)自动绑定forward方法
class CenteredLayer(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()


# 自定义带优化参数的层：
class MyDense(gluon.nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(units, in_units, **kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))
    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


# 2. 利用gluon.nn.Block自定义Sequential对象
class SequentialDemon(gluon.nn.Block):
    """gluon.nn.Sequential的简单含义：Block的子类，存储子Block对象的容器"""
    def __init__(self, **kwargs):  # 传入实参应为字典或键值对
        super(SequentialDemon, self).__init__(**kwargs)  # 调用父类构造方法

    # 将block添加入成员变量（list)  ***error: 运行提示OrderedDict没有append方法
    def add(self, block):
        self._children.append(block)

    def forward(self, X):
        for block in self._children:
            X = block(X)
        return X

# 实例化
Model = SequentialDemon()
with Model.name_scope():
    Model.add(gluon.nn.Dense(16, activation='relu'))
    Model.add(gluon.nn.Dense(2))
Model.initialize()


# 3. Block和Sequential嵌套
class RecMLP(gluon.nn.Block):  # recurisive嵌套
    """初始化时，Block内嵌套net对象"""
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = gluon.nn.Sequential()
        with self.name_scope():
            self.net.add(gluon.nn.Dense(256, activation='relu'))
            self.net.add(gluon.nn.Dense(128, activation='relu'))
            self.dense = gluon.nn.Dense(8)

    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))

"""*******************************************************************************************************************
Section_2: 模型initialize()的意义
要点：
    1. 延迟初始化：
        1）未指定输入层个数时，initialize时，权重、偏置等暂不初始化，如：gluon.nn.Dense(16)，待传入数据时再参数初始化，如：net(X)；
        2）指定输入层个数时，initialize时，权重、偏置等便初始化，如：gluon.nn.Dense(16, in_units=8)。
    2. 可自定义模型初始化：        
"""


"""*******************************************************************************************************************
Section_3: 数据存盘和模型参数存盘：nd.save()/nd.load()以及net.save_params()/net.load_params()使用
"""

file_path = '/home/frank/MXNet_Learning/Lesson_3/nd_save_demon.params'
file_name = 'model.params'

# 数据的存取
nd.save(file_name, x)
nd.save('x.txt', x)

# 模型参数的存取
model = gluon.nn.Dense()
model.save_params(file_name)
model.load_params(file_name)


"""*******************************************************************************************************************
Section_4: dropout正则化及gluon实现
含义：
    以一定概率，将某层输出X中的某些元素置为0（即：屏蔽掉某些神经元的输出）
"""


# dropout正则化方法实现
def dropout(X, drop_prob):
    keep_prob = 1 - drop_prob
    assert 0 <= keep_prob <= 1, 'prob must be in (0, 1)'
    if keep_prob == 0:
        return X.zeros_like()
    # mask 用以屏蔽某条件的元素
    mask = nd.random.uniform(0,1, shape=X.shape) < keep_prob
    # scale用于保证E(X)不变，drop掉某些点后，E(X)后减小
    scale = 1 / keep_prob
    return X * mask * scale   


# gluon实现dropout正则化
model = gluon.nn.Sequential()
with model.name_scope():
    model.add(gluon.nn.Dense(16, activation='relu'))
    model.add(gluon.nn.Dropout(0.5))     # 直接调用gluon.nn的Dropout()对象，实现dropout
    model.add(gluon.nn.Dense(1))




