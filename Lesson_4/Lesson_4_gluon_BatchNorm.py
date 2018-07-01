from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    # 卷积后, 激活前, 进行批量归一化
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=15))
    net.add(gluon.nn.BatchNorm(axis=1))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

    net.add(gluon.nn.Conv2D(channels=30, kernel_size=3))
    net.add(gluon.nn.BatchNorm(axis=1))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

    # Dense层前, 扁平化处理
    net.add(gluon.nn.Flatten())

    # Dense层
    net.add(gluon.nn.Dense(128, activation='relu'))
    net.add(gluon.nn.Dense(10))
