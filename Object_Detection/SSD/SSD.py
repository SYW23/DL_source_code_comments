# 源码地址 https://github.com/amdegroot/ssd.pytorch
# 阅读顺序大抵从下往上（以块为单位）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers    融合层部分（头部网络）
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        # 先验框个数 38x38x4+19x19x6+10x10x6+5x5x6+3x3x4+1x1x4=8732
        self.priorbox = PriorBox(self.cfg)    # 返回的shape为[8732,4]
        # volatile=True的节点不会求导，也不会进行反向传播
        # 对于不需要反向传播的情景(测试推断)，该参数可以实现一定速度的提升，并节省一半的显存，因为其不需要保存梯度
        # 该属性已在0.4版本中被移除，可以使用with torch.no_grad()代替该功能
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):    # 前向
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()    # 特征图
        loc = list()    # 特征图经过回归后的信息
        conf = list()    # 特征图经过分类后的信息
        
        # 主干VGG网络
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        
        # 对conv4_3进行L2正则化
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        
        # 新增层
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        
        # 融合层
        # apply multibox head to source layers
        # l(x)和c(x)经过permute从[batch_size,channel,height,weight]变为[batch_size,height,weight,channel]，方便后续处理
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # contiguous()开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # 利用view函数进行格式变换，保留batch_size维度，剩余维度合并（-1代表未知，由程序自己推算）
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)    # 变为[batch_size,34928]，34928=8732*4
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)    # 同理，变为[batch_size,8732*21]
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc predicts
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf predicts，需要经过softmax
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            # 再次变换格式，为计算loss做准备
            output = (
                loc.view(loc.size(0), -1, 4),    # [batch_size,num_priors,4]
                conf.view(conf.size(0), -1, self.num_classes),    # [batch_size,num_priors,类别数]
                self.priors    # [num_priors,4]
            )
        return output

    def load_weights(self, base_file):    # 加载权重文件
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,    # load_state_dict的操作对象是一个具体的对象,而不能是文件名
                                 map_location=lambda storage, loc: storage))    # 将gpu训练好的模型参数load到cpu上
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i    # 输入通道数
    
    # 主体部分（到conv5_3为止）
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)    # 空洞卷积
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False    # flag用于控制交替使用1x1和3x3的卷积核
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):    # 传入vgg主干网络list和新增网络list
    loc_layers = []    # 用于推测框的位置（回归）
    conf_layers = []    # 用于推测框的置信度（分类）
    vgg_source = [21, -2]    # 前者为conv4_3的索引，后者为conv7的索引
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,    # vgg[v]提取需要融合的特征图
                                 cfg[k] * 4, kernel_size=3, padding=1)]    # cfg[k]是特征图中每个点对应anchor的数量，*4表示anchor的四个坐标
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]    # *num_classes表示每个anchor对应每个类别的置信度
    for k, v in enumerate(extra_layers[1::2], 2):    # 从索引1开始，2为步长，k从2开始计数（递增1）
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


# 网络参数定义
base = {    # 主干网络（VGG16-Conv5_3）参数，'M'表示maxpooling层，'C'表示maxpooling层（ceil_mode=True）
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {    # 新增层参数，'S'表示stride=2的降采样
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {    # 特征融合层参数，数字表示每层特征图中每个点对应anchor的数量
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


# 构建SSD
def build_ssd(phase, size=300, num_classes=21):    # SSD300，类别数20+1（背景）
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),    # 主干
                                     add_extras(extras[str(size)], 1024),    # 新增
                                     mbox[str(size)], num_classes)    # 融合
    return SSD(phase, size, base_, extras_, head_, num_classes)
