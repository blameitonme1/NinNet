import torch
from torch import nn
from d2l import torch as d2l
# 定义nin块，一个3x3卷积加上两个1x1卷积，相当于两个全连接层，显著减少了参数数量
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        # 核心就是加入了两层的一层卷积，等于是一个全连接层（作用在所有像素上，所有的像素分享同样的权重），参数没那么多，
        nn.Conv2d(out_channels, out_channels, kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2), # 用来保留显著特征，减少图片尺寸
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1 ,padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # fashionMnist, 10个种类
    nin_block(384, 10, kernel_size=3, strides=1 ,padding=1),
    # 全局平均池化层
    nn.AdaptiveAvgPool2d((1,1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten()
)
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
