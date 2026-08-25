# -*- coding: utf-8 -*-
"""
model.py —— 模型定义

本文件只回答一个问题：“模型长什么样”，不负责训练。
目前包含：
  1. SimpleCNN：从零实现的简单卷积神经网络（方法一）
  2. ResNet18/34/50/101/152：加载 ImageNet 预训练权重后做迁移学习

读懂本文件需要的基础概念：
  - 一张 32x32 彩色图在 PyTorch 里是形状 (3, 32, 32) 的张量
  - 一个 batch 是 (N, 3, 32, 32)，N 是一批有多少张图
  - 卷积层用“小窗口扫过整张图”的方式提取局部特征
  - 池化层把图缩小，保留主要信息
  - 全连接层把前面的特征汇总成“每个类别的得分”
"""

import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


class SimpleCNN(nn.Module):
    """方法一：从零实现的简单 CNN。

    结构：3 组“卷积块”提取特征 + 1 个全连接层做分类。
    每一组卷积块 = 卷积(提取特征) + 批归一化(加速收敛) + ReLU(非线性) + 池化(缩小尺寸)。

    参数：
        num_classes : int，输出类别数
        dropout     : float，分类层前的 Dropout 概率，0 表示不使用
    """

    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()

        self.features = nn.Sequential(
            # 第 1 组：输入 (3, 32, 32) -> 输出 (32, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 2 组：输入 (32, 16, 16) -> 输出 (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 3 组：输入 (64, 8, 8) -> 输出 (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dropout 只在分类层前使用。dropout=0 时不创建 Dropout，
        # 这样旧 checkpoint 的 state_dict 仍然可以兼容。
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # 展平后是 128 * 4 * 4 = 2048 维，再映射到 num_classes 个类别得分
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        """前向传播：输入图片 batch，输出每个类别的得分。"""
        x = self.features(x)
        x = x.flatten(1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


RESNET_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
}


def create_resnet(backbone="resnet18", num_classes=10, dropout=0.0):
    """方法二：ImageNet 预训练的 ResNet + 迁移学习。

    迁移学习的思想：
    ResNet 已经在 ImageNet（1000 类、上百万张图）上学会了“怎么看一张图”的通用能力，
    我们把它学好的大部分参数直接拿来用，只修改输入/输出结构以适应 CIFAR-10。

    CIFAR-10 只有 32x32，因此要对原版 ResNet 做三处小修改：
      1) 第一个 7x7、步长 2 的卷积，改成 3x3、步长 1、padding 1；
      2) 去掉第一个卷积后的 maxpool；
      3) 最后的全连接层从 1000 类改成 10 类。

    参数：
        backbone    : str，resnet18/resnet34/resnet50/resnet101/resnet152
        num_classes : int，输出类别数
        dropout     : float，分类层前的 Dropout 概率
    """
    if backbone not in RESNET_BUILDERS:
        raise ValueError(f"未知 ResNet: {backbone}")

    builder, weights = RESNET_BUILDERS[backbone]
    model = builder(weights=weights)

    # 1) 为大图设计的 7x7 卷积，改成适合 32x32 小图的 3x3 卷积
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 2) 原版第一个卷积后还有最大池化，会把 32x32 再缩小一半；CIFAR-10 太小，直接去掉
    model.maxpool = nn.Identity()

    # 3) 替换最后的分类层
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def create_model(model_name, num_classes=10, dropout=0.0):
    """模型工厂：根据名字返回对应模型，供 train.py / test.py 共用。

    兼容旧的“resnet”写法：resnet 会按 resnet18 处理。
    """
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    if model_name == "resnet":
        model_name = "resnet18"

    if model_name in RESNET_BUILDERS:
        return create_resnet(
            backbone=model_name,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(
        f"未知模型名: {model_name}，"
        "可选 'cnn'、'resnet18'、'resnet34'、'resnet50'、'resnet101'、'resnet152'"
    )
