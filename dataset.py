# -*- coding: utf-8 -*-
"""
dataset.py —— CIFAR-10 数据加载模块

作用：把 CIFAR-10 数据集从磁盘加载进来，做好预处理（数据增强、转张量、
标准化），最后封装成 PyTorch 的 DataLoader 供训练和测试使用。

为什么需要这个模块？
深度学习代码一般按职责拆成几个文件：
  - dataset.py：负责"数据怎么进来"（本文件）
  - model.py：负责"模型长什么样"
  - train.py：负责"怎么训练"
  - test.py：负责"怎么测试"
  - utils.py：存放各种小工具函数
这样每个文件只干一件事，代码清晰、可复用，也方便排查问题。
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 数据集的官方统计值：三通道(R,G,B)的均值(mean)和标准差(std)
# 这两个常数是论文/官方数据里给出的，直接拿来用即可。
# 标准化的目的：把像素值从 [0, 1] 范围"压"到均值为 0、标准差为 1 的分布，
# 让不同通道的数值尺度一致。这样可以加速模型收敛，也让训练过程更稳定。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_dataloader(batch_size, data_root):
    """
    返回 CIFAR-10 的训练集 DataLoader 和测试集 DataLoader。

    参数：
        batch_size : int，每个 batch 里放多少张图片
        data_root  : str，数据存放/下载的目录路径

    返回：
        (train_loader, test_loader) 两个 DataLoader 对象
    """

    # ---------------- 训练集预处理 ----------------
    # transforms.Compose 把一系列预处理操作按顺序组合起来，
    # 每个样本经过这个流水线后，才会被送入模型。
    transform_train = transforms.Compose([
        # 1. 随机水平翻转：有 50% 的概率把图片左右镜像。
        #    这是一种"数据增强"(data augmentation)手段：CIFAR-10 里像"汽车、
        #    飞机"这类物体左右翻转后仍然是合法的图片，所以模型可以用同一张
        #    图学到两种形态，相当于免费扩充了训练数据，能有效抑制过拟合，
        #    提升模型泛化能力（在新图片上的表现）。
        transforms.RandomHorizontalFlip(),

        # 2. 随机裁剪：先把 32x32 的图片四周各填充 4 个像素（变成 40x40），
        #    再从中随机裁剪出一块 32x32 的区域。
        #    这样每次看到的是"图片略微平移"后的版本，同样是为了增加数据
        #    多样性，让模型对物体的位置不敏感，学到的特征更鲁棒。
        transforms.RandomCrop(32, padding=4),

        # 3. 转成张量：PIL 图片 -> PyTorch Tensor，并把像素值从 [0,255]
        #    缩放到 [0,1]（ToTensor 自动完成除以 255），同时把通道维度
        #    排到前面，形状从 (H, W, C) 变成 (C, H, W)。
        #    这是 PyTorch 卷积层要求的输入格式，例如 (3, 32, 32)。
        transforms.ToTensor(),

        # 4. 标准化：用官方均值和标准差把每个通道归一化到近似 N(0,1) 分布，
        #    让模型训练更稳定、收敛更快（理由见文件开头的注释）。
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # ---------------- 测试集预处理 ----------------
    # 测试集只做"必要"的转换：转张量 + 标准化。
    # 注意：测试集绝对不能做随机增强！因为增强是"故意引入随机性"来增加
    # 训练数据多样性；如果测试时也随机裁剪/翻转，同一个测试样本每次进去
    # 结果都不一样，评价指标就不稳定，也无法公平地反映模型真实水平。
    # 测试集要做的只是把图片变成模型能吃的格式，并保持和训练集一致的数据
    # 分布（标准化参数必须和训练集完全相同）。
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # ---------------- 加载数据集 ----------------
    # datasets.CIFAR10 会先去 data_root 里找数据；如果不存在，
    # download=True 会自动下载到该目录（下载一次后本地就有缓存了）。
    # train=True 表示训练集（50000 张），train=False 表示测试集（10000 张）。
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    # ---------------- 封装成 DataLoader ----------------
    # DataLoader 负责：把数据集按 batch_size 切分成小批次、每个 epoch 打乱
    # 顺序、用多个子进程并行读数据（num_workers），从而高效地喂给模型。
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,      # 训练集必须打乱：让每个 epoch 看到的数据顺序不同，
                           # 避免模型"记住"固定顺序而产生偏差，也有助于收敛
        num_workers=2,     # 用 2 个并行子进程读数据，加快训练
        drop_last=False,   # 最后不足一个 batch 的尾巴也保留（50000 能整除很多
                           # batch_size，但保留尾巴更稳妥，不会丢数据）
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,     # 测试集不打乱：评估结果要稳定可复现
        num_workers=2,
        drop_last=False,
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # 独立运行本文件时执行这里的测试代码：
    #   python dataset.py
    # 用来验证数据加载、预处理、DataLoader 封装是否正常。

    train_loader, test_loader = get_cifar10_dataloader(
        batch_size=8, data_root='./data'
    )

    # 打印数据集规模信息
    print(f'训练集样本数: {len(train_loader.dataset)}')
    print(f'测试集样本数: {len(test_loader.dataset)}')
    print(f'类别数: {len(train_loader.dataset.classes)}')
    print(f'类别: {train_loader.dataset.classes}')

    # 从训练 DataLoader 里取一个 batch，验证形状是否正确
    images, labels = next(iter(train_loader))
    print('\n--- 训练集一个 batch 的信息 ---')
    print(f'图片张量形状 (batch, 通道, 高, 宽): {tuple(images.shape)}')
    print(f'标签张量形状 (batch,): {tuple(labels.shape)}')
    print(f'图片元素类型: {images.dtype}，标签元素类型: {labels.dtype}')
    print(f'图片数值范围（标准化后，均值应接近0）: '
          f'min={images.min():.3f}, max={images.max():.3f}, '
          f'mean={images.mean():.3f}')
    print(f'本 batch 的标签: {labels.tolist()}')

    # 再验证一下测试集也能正常取数据（形状应与训练集一致）
    test_images, test_labels = next(iter(test_loader))
    print('\n--- 测试集一个 batch 的信息 ---')
    print(f'图片张量形状: {tuple(test_images.shape)}')
    print(f'标签张量形状: {tuple(test_labels.shape)}')

    print('\n数据加载验证通过！')
