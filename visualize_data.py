# -*- coding: utf-8 -*-
"""
visualize_data.py —— 数据可视化练习

目标：把 CIFAR-10 训练集的一个 batch 画成图片网格，
亲眼看看模型"看到"的数据长什么样。

写代码时想清楚这几个问题：
1. 为什么要"反标准化"才能正常显示？（标准化做了 (x - mean) / std，显示前要还原）
2. permute(1, 2, 0) 在干什么？（把 (C, H, W) 转成 matplotlib 需要的 (H, W, C)）
3. 标签是整数，怎么显示成类别名字？（dataset.classes 列表）

运行：
    C:\Users\shisan\miniconda3\envs\cifar\python.exe visualize_data.py

参考骨架（练习用，自己补齐每一行并理解）：

    import matplotlib.pyplot as plt
    import torch
    from dataset import get_cifar10_dataloader

    # 1. 取一个 batch（25 张图）
    train_loader, _ = get_cifar10_dataloader(batch_size=25, data_root='./data')
    images, labels = next(iter(train_loader))

    # 2. 反标准化：mean/std 要变成 (3, 1, 1) 的形状才能和 (N, 3, H, W) 广播运算
    # 3. clamp 到 0~1，避免显示时出现异常颜色
    # 4. 用 plt.subplots 建 5x5 的画布，逐张 imshow
    # 5. 每张图标题写类别名，关掉坐标轴，最后 plt.show()
"""
import matplotlib.pyplot as plt
import torch
from dataset import get_cifar10_dataloader

