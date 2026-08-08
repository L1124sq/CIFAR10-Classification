# -*- coding: utf-8 -*-
"""
model.py —— 模型定义

本文件只回答一个问题："模型长什么样"，不负责训练。
包含两种模型：
  1. SimpleCNN：从零实现的简单卷积神经网络（方法一）
  2. ResNet18：加载 ImageNet 预训练权重后做迁移学习（方法二）

读懂本文件需要的基础概念：
  - 一张 32x32 彩色图在 PyTorch 里是形状 (3, 32, 32) 的张量（C=通道, H=高, W=宽）
  - 一个 batch 是 (N, 3, 32, 32)，N 是一批有多少张图
  - 卷积层用"小窗口扫过整张图"的方式提取局部特征
  - 池化层把图缩小一半，保留主要信息
  - 全连接层把前面的特征汇总成"每个类别的得分"
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """方法一：从零实现的简单 CNN。

    结构：3 组"卷积块"提取特征 + 1 个全连接层做分类。
    每一组卷积块 = 卷积(提取特征) + 批归一化(加速收敛) + ReLU(非线性) + 池化(缩小尺寸)。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # ---------------- 特征提取部分 ----------------
        # nn.Sequential：按顺序依次执行里面的每一层，前一层输出就是后一层输入
        self.features = nn.Sequential(
            # 第 1 组：输入 (3, 32, 32) -> 输出 (32, 32, 32)
            # Conv2d(输入通道数, 输出通道数, 卷积核大小, padding=1)
            #   - 输入通道 3 = RGB 三个颜色
            #   - 输出通道 32 = 用 32 个卷积核，提取 32 种不同的局部特征
            #   - padding=1 保证 32x32 图经过 3x3 卷积后尺寸不变
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # BatchNorm2d：把这一层的输出分布拉回到标准范围，让训练更稳定、收敛更快
            nn.BatchNorm2d(32),
            # ReLU：把负数变成 0。没有它，多层线性运算叠起来还是线性，学不了复杂规律
            nn.ReLU(),
            # MaxPool2d(2)：取 2x2 窗口里的最大值，尺寸减半 32x32 -> 16x16
            nn.MaxPool2d(2),

            # 第 2 组：输入 (32, 16, 16) -> 卷积 -> (64, 16, 16) -> 池化 -> (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 3 组：输入 (64, 8, 8) -> 卷积 -> (128, 8, 8) -> 池化 -> (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ---------------- 分类部分 ----------------
        # 到这一步特征图是 (128, 4, 4)：128 个通道，每个 4x4
        # 展平成 128*4*4 = 2048 个数字后，用全连接层映射成 10 个类别的得分
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        """前向传播：输入一个 batch 的图片，输出每个类别的得分。

        形状变化：x (N, 3, 32, 32)
                -> features: (N, 128, 4, 4)
                -> flatten:  (N, 2048)
                -> classifier: (N, 10)
        """
        x = self.features(x)     # 特征提取
        x = x.flatten(1)         # 展平：从第 1 维（通道）开始压平，(N,128,4,4) -> (N,2048)
        x = self.classifier(x)   # 分类：2048 个特征 -> 10 个类别得分
        return x


def create_resnet18(num_classes=10):
    """方法二：ImageNet 预训练的 ResNet18 + 迁移学习。

    迁移学习的思想：ResNet18 已经在 ImageNet（1000 类、上百万张图）上学会了
    "怎么看一张图"的通用能力（边缘、纹理、形状等特征），我们把它学好的
    大部分参数直接拿来用，只需要改最后一层，让它输出我们自己的 10 个类别。

    注意：ImageNet 的图是 224x224，而 CIFAR-10 只有 32x32，所以要做 3 处小修改：
    """
    # 加载在 ImageNet 上预训练好的 ResNet18 权重（第一次运行会自动下载权重文件）
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 1) 原版第一个卷积层是 7x7、步长 2（为大图设计），换成 3x3、步长 1，
    #    这样 32x32 的输入可以直接流过网络而不至于尺寸缩得太快
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 2) 原版第一个卷积后还有个最大池化会把尺寸再减半，32x32 太小了，直接去掉
    #    nn.Identity() 是"什么都不做"的层，相当于把这一步跳过
    model.maxpool = nn.Identity()

    # 3) 原版最后一层输出 1000 类（ImageNet 的类别数），换成输出 10 类
    #    model.fc.in_features 是上一层给它的特征数（512），保持不动
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def create_model(model_name, num_classes=10):
    """模型工厂：根据名字返回对应的模型，让 train.py / test.py 共用一套代码。"""
    if model_name == 'cnn':
        return SimpleCNN(num_classes=num_classes)
    if model_name == 'resnet':
        return create_resnet18(num_classes=num_classes)
    raise ValueError(f"未知模型名: {model_name}，可选 'cnn' 或 'resnet'")
