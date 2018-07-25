#!/usr/bin/env python
# coding:utf-8
"""
采用SSD算法, 进行皮卡丘检测
"""

import time
import numpy as np

from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet import autograd
from mxnet import metric
from mxnet import init
from mxnet import gpu, cpu
from matplotlib import pyplot as plt
from mxnet.gluon import nn
from mxnet.ndarray.contrib import MultiBoxPrior
from mxnet.ndarray.contrib import MultiBoxTarget
from mxnet.ndarray.contrib import MultiBoxDetection


"""
绘图, 绘边框函数
"""
def box_to_rect(box, color, linewidth=3):
    """ 根据传入坐标, 颜色和线宽, 绘制边框 """
    box = box.asnumpy()
    return plt.Rectangle(
        xy=(box[0], box[1]),
        width=box[2] - box[0],
        height=box[3] - box[1],
        fill = False,
        edgecolor=color,
        linewidth=linewidth
    )


def plt_box(batch, nrows=3, ncols=3, img_shape=256):
    """  绘制图片, 并绘制图片中物体检测的边框  """
    _, figs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 24))
    for row in range(nrows):
        for col in range(ncols):
            """
            batch为io.DataBatch类对象, .data和.label为类成员变量;
            为保证调用next()方法, 构造io.DataBatch时, data和label为list对象, 加了[data]和[label]作为实参传入, 故需data[0]和label[0]来取出数据            
            """
            # img: height x width x channel;
            # label: num_boxs x 5;  5维向量: [class_label, box_x_min, box_y_min, box_x_max, box_y_max]
            img, labels = batch.data[0][ncols * row + col], batch.label[0][ncols * row + col]

            # 将图样数据转为numpy()格式, 并调整至(0, 255)内
            img = img.transpose((1, 2, 0)) + rgb_mean
            img = img.clip(0, 255).asnumpy() / 255      # 截断元素至(0, 255)内

            # 绘制图像
            fig = figs[row][col]
            fig.imshow(img)
            # 绘制边框
            for label in labels:
                rect = box_to_rect(label[1:5] * img_shape, 'red', 2)  # box原始坐标为(0,1)之间, 绘图时应 × img_shape
                fig.add_patch(rect)
            # 坐标轴不显示
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
    plt.show()


def plt_anchor_box(px, py, img_size, boxes):
    """  绘制传入坐标px, py作为锚点的所有边框  """
    colors = ['blue', 'green', 'red', 'black', 'magenta']
    plt.imshow(nd.ones((img_shape, img_shape, 3)).asnumpy())        # ones为白色, 0.5为灰色, zeros为黑色
    # 取出锚点所有box
    anchor = boxes[px, py, :, :]
    # 遍历锚点每个边框
    for i in range(anchor.shape[0]):
        box = box_to_rect(anchor[i,:] * img_size, colors[i])
        plt.gca().add_patch(box)
    plt.show()



"""**************************************************************************************************************
Section 1: 准备数据集
    1. 下载数据集;
    2. 设置初始化参数: imag_shape, batch_size,
    3. 获取数据集迭代器, 类别数: train_data, test_data 以及num_class
    4. 获取一个train_data的batch
           
    注意点:
        1. batch.data[0]和batch.label[0] 数据取法:
            batch是一个io.DataBatch类对象, 构造时为保证传入实参为list, 添加了[data], 故: 取数据时, 需用.data[0]和.label[0]
        2. data和label的shape:
            batch的label的shape: batch_size x num_box x 5
            每个边框的label为一个长为5的数组: 
            1) 第一个元素是其对用物体的标号，其中-1表示非法物体，仅做填充使用;
            2) 后面4个元素表示边框坐标: 前两位左下角x/y, 后两位右上角x/y.   
"""


def download_data():
    """ 下载皮卡丘数据 """
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/')
    data_dir = '../data/pikachu/'
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
              'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
              'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for key, value in dataset.items():
        f_path = gluon.utils.download(root_url+key, path=data_dir+key, sha1_hash=value)
        print('file path is: ', f_path)
    return data_dir


def get_iterators(data_shape, batch_size, data_dir):
    """  获取训练集和验证集迭代器  """
    # lable名和label数
    class_names = ['pikachu']
    num_classes = len(class_names)

    # 训练集迭代器
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir+'train.rec',
        path_imgidx=data_dir+'train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200
    )

    # 验证集迭代器
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir+'val.rec',
        shuffle=False,
        mean=True,
    )
    return train_iter, val_iter, class_names, num_classes


# 初试参数
img_shape = 256
batch_size = 32
rgb_mean = nd.array([123, 117, 104])
num_classes = 1
num_epochs = 30

# 锚框大小: 相较于imag_size的scale
sizes=[0.5, 0.25, 0.1]
# 锚框的长宽比
ratios=[1, 2, 0.5]
# 每个锚点的锚框数
num_anchors = len(sizes) + len(ratios) - 1

# 下载数据集
data_dir = download_data()
# 获取训练集和验证集迭代器
train_data, test_data, class_names, num_classes = get_iterators(img_shape, batch_size=batch_size, data_dir=data_dir)
# 获取一个train_data的batch
batch = train_data.next()


"""************************************************************************************************************
Section 2: 构建SSD模型
  1. 模型主线: 主体网络Body() + 减半模块() + 全局池化层;
  2. 模型旁路: 类别预测模块 + 边框预测模块;
  3. 采用两个预测模块, 分别对模型主线模块进行预测, 最后结果concat在一起
"""

# 每个像素点作为锚点, 均生成: (sizes + ratios -1)个锚框box
# box_raw = MultiBoxPrior(img, sizes=sizes, ratios=ratios)
# 将锚框reshape成一个以像素点为长/宽, 以每个像素的锚框矩阵为元素的矩阵, 每个像素的box数 x 4(每个边框需4个坐标确定)
# boxes = bow_raw.reshape(shape=(img_size, img_size, -1, 4))


def body_net():
    """  主体网络, 可选用ResNet等经典网络, 用以对原始图像做卷积运算, 抽取feature map  """
    out = nn.HybridSequential()
    for num_filters in (16, 32, 64):
        out.add(down_sample(num_filters=num_filters))
    return out


def down_sample(num_filters):
    """  将图像尺寸减半  """
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(channels=num_filters, kernel_size=3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation(activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2))
    return out


def class_predictor(num_anchors, num_classes):
    """ 图样分类卷积层: 返回一个卷积层用以预测锚框分类 """
    # channels数: 锚框数 x (类别数 + 1), 1为背景
    return nn.Conv2D(channels=num_anchors*(num_classes+1), kernel_size=3, padding=1)


def box_predictor(num_anchors):
    """  边框回归卷积层: 返回一个卷积层用于预测边框位置  """
    # channels数: 每个锚点的锚框数 x 4, 4为每个锚框的坐标个数
    return nn.Conv2D(channels=num_anchors*4, kernel_size=3, padding=1)


def flatten_prediction(pred):
    """
    转换过程:
    1) 输入的pred(class_pred 或 box_pred): batch_size x channel x height x width;
    2) 先transpose成: batch_size x height x width x channel;
    3) 再flatten成(2, height * width * channel)的2-D矩阵;
    """
    return pred.transpose(axes=(0, 2, 3, 1)).flatten()          #TODO: flatten后的数据的shape时2-D还是1-D?


def concat_prediction(*preds):
    """  将多层网络的输出, 作为多个实参传入, 参入的数据为2-D的(batch_size, h * w * c), 按照dim=1轴concat在一起"""
    return nd.concat(*preds, dim=1)


def toy_ssd_model(num_anchors, num_classes):
    """  构建玩具SSD模型 """
    # 减半模块
    down_samples = nn.Sequential()
    for _ in range(3):
        down_samples.add(down_sample(num_filters=128))

    # 预测模块: 类别预测和边框预测
    class_predictors = nn.Sequential()
    box_predictors = nn.Sequential()
    for _ in range(5):  # 原图/3个减半模块/最后全局池化层, 共需5个预测
        class_predictors.add(class_predictor(num_anchors=num_anchors, num_classes=num_classes))
        box_predictors.add(box_predictor(num_anchors=num_anchors))

    # model模型: 主体网络, 下降模块, 类别预测模块, 边框预测模块
    model = nn.Sequential()
    model.add(body_net(), down_samples, class_predictors, box_predictors)
    return model


def toy_ssd_forward(x, model, sizes, ratios, verbose=False):
    """  定义一个前向传播函数  """
    # 拿到主体网络, 下降模块, 类别预测模块, 边框预测模块
    body, down_samples, class_predictors, box_predictors = model
    # 定义锚点, 类别预测, 边框预测
    anchors, class_preds, box_preds = [], [], []

    # feature extraction
    x = body(x)

    # 5层预测: 原始图像 + 3层减半模块 + 全局Pool池化
    for i in range(5):
        # anchor shape: batch_size x 每幅图像锚框数 x 4, return时, 沿dim = 1 concat.
        anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        # class_predictor返回的shape为: batch_size x  channel x height x width, 而后flatten成:batch_sizse x (height x width x channel), return时再concat一起
        class_preds.append(flatten_prediction(class_predictors[i](x)))
        # 同class
        box_preds.append(flatten_prediction(box_predictors[i](x)))

        if verbose:
            print('Predict scale', i, x.shape, 'with', anchors[-1].shape[1], 'anchors')

        if i < 3:
            x = down_samples[i](x)
        elif i == 3:
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(x.shape[2], x.shape[3]))

    # concat data
    return (concat_prediction(*anchors),          # anchors: batch_size x (dim=1, 每层输出concat一起) x 4
            concat_prediction(*class_preds),      # class_preds: batch_size x (dim=1, 每层输出的h * w * c, concat一起)
            concat_prediction(*box_preds))


class ToySSD(gluon.Block):
    """  定义一个简单的SSD类对象
    返回值:
        anchors: (batch,  (),  4)
        class_preds:  (batch, (), num_classes+1)
        box_preds:  (batch, ())
    """
    def __init__(self, num_classes, verbose=True, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor box sizes and ratios for 5 feature scales
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88], 0.961]  # 图像逐渐变小, 锚框范围逐渐便大
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_classes = num_classes
        self.verbose = verbose
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        # use name_scope to guard the names
        with self.name_scope():
            self.model = toy_ssd_model(num_anchors, num_classes)

    def forward(self, x):
        anchors, class_preds, box_preds = toy_ssd_forward(x, self.model, self.sizes, self.ratios, verbose=self.verbose)
        # it it better to have class predictions reshaped for softmax coumpute
        class_preds = class_preds.reshape(shape=(0, -1, self.num_classes+1))        # class_preds 被reshape成为: batch_size x () x (num_classes+1)
        return anchors ,class_preds, box_preds

"""************************************************************************************************************
Section 3: 训练模型
  1. 定义损失函数;
  2. 定义优化器;
  3. 获取SSD模型;
  4. 训练, 优化求解
"""


def training_targets(anchors, class_preds, labels):
    """
    输入:
        anchors is:  (1, 5444, 4)
        class_preds is:  (32, 5444, 2)
        labels is:  (32, 3, 5)

    返回: 3个NDArray:
        1) 预测的边框跟真实边框的偏移，大小是batch_size x (num_anchors*4)
        2) 用来遮掩不需要的负类锚框的掩码，大小跟上面一致
        3) 锚框的真实的标号，大小是batch_size x num_anchors
    """
    class_preds = class_preds.transpose(axes=(0, 2, 1))
    return MultiBoxTarget(anchors, labels, class_preds)


# 定义损失函数
def focal_loss(gamma, p):
    """  返回一个负类项带有gamma次方的交叉熵  """
    return -(1-p) ** gamma * np.log(p)   # p为正类概率
class FocalLoss(gluon.loss.Loss):
    """  定义一个损失类, 使的在交叉熵损失的负类项带有gamma次方  """
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pj) ** self._gamma) * pj.log()
        return loss.mean(axis=self._batch_axis, exclude=True)


class SmoothL1Loss(gluon.loss.Loss):
    """  定义一个损失类, 用于L1损失在0点出顺滑  """
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)

def get_ctx():
    try:
        ctx = nd.zeros(shape=(1,), ctx=gpu())
        ctx = gpu()
    except:
        ctx = cpu()
    return ctx
ctx = get_ctx()


cls_loss = FocalLoss()
box_loss = SmoothL1Loss()

cls_metric = metric.Accuracy()
box_metric = metric.MAE()

# the CUDA implementation requres each image has at least 3 labels, Padd two -1 labels for each instance
train_data.reshape(label_shape=(3, 5))
train_data = test_data.sync_label_shape(train_data)


# 获取模型
net = ToySSD(num_classes)
net.initialize(init.Xavier(magnitude=2), ctx=ctx)

# 获取优化器
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

# 迭代epochs次
for epoch in range(num_epochs):
    """  reset data iterators and metircs for each epoch  """
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()

    tic = time.time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)

        with autograd.record():
            anchors, class_preds, box_preds = net(x)
            # 将锚框、预测类、真实类传入
            box_target, box_mask, cls_target = training_targets(anchors, class_preds, y)
            # 损失函数
            loss_1 = cls_loss(class_preds, cls_target)  # 传入预测值、真实值
            loss_2 = box_loss(box_preds, box_target, box_mask)   # 传入预测值、真实值、掩码
            loss = loss_1 + loss_2      # 可加权求和
        loss.backward()
        trainer.step(batch_size)

        # update metrics
        cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
        box_metric.update([box_target], [box_preds * box_mask])     # 采用掩码, 对预测值进行屏蔽

    print('Epoch %2d, train %s %0.2f, %s %0.5f, time %0.1f sec' %  (epoch, *cls_metric.get(), *box_metric.get(), time.time()-tic))

"""************************************************************************************************************
Section 4: 预测
  1. 定义损失函数;
  2. 定义优化器;
  3. 获取SSD模型;
  4. 训练, 优化求解
"""

def process_image(fname):
    """  图像读取及预处理函数, 返回一个batch x channel x height x width矩阵  """
    with open(fname, 'rb') as f:
        im = image.imcode(f.read())
    # resize to data_shape
    data = image.imresize(im, data_shape, data_shape)
    # minus rgb mean
    data = data.astype('float32') - rgb_mean
    # convert to batch x channel x height x width
    return data.transpose((2,0,1)).expand_dims(axis=o), im


def predict(x):
    """  预测函数会输出所有边框，每个边框由[class_id, confidence, xmin, ymin, xmax, ymax]  """
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(cls_preds.transpose(0,2,1), mode='channel')
    return MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=True, clip=False)


def display(im, out, threshold=0.5):
    """  绘制出超过阀值的边框  """
    plt.imshow(im.asnumpy())
    for row in out:
        row = row.asnumpy()
        class_id, score = int(row[0], row[1])
        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id % len(colors)]
        box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().text(box[0], box[1], '{:s} {:0.2f}'.format(text, score), bbox = dict(facecolor=color, alpha=0.5), fontsize=10, color='white')
    plt.show()


x, im = process_image('../img/pikachu.jpg')
out = predict(x)
print('out.shpae is: ', out.shape)

display(im, out[0], threshold=0.5)

