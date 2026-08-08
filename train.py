# -*- coding: utf-8 -*-
"""
train.py —— 训练主程序

整段代码回答一个问题："怎么让模型学会分类？"
答案就一句话：反复执行"前向算出预测 -> 算损失 -> 反向传播算梯度 -> 更新参数"，
每个 epoch 把 50000 张训练图完整过一遍，然后到验证集上检查一次水平。

运行示例：
    python train.py --model cnn    --epochs 30
    python train.py --model resnet --epochs 15 --lr 1e-3
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import get_cifar10_dataloader
from model import create_model
from utils import set_seed, get_device, save_checkpoint, evaluate, plot_curves


def parse_args():
    """命令行参数：不写死的配置都放这里，换参数不用改代码。"""
    parser = argparse.ArgumentParser(description='CIFAR-10 分类训练')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                        help='选择模型：cnn 或 resnet')
    parser.add_argument('--epochs', type=int, default=30, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=128, help='每批多少张图')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率（参数更新的步长）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='从训练集切多少比例当验证集')
    parser.add_argument('--data_root', type=str, default='./data', help='数据目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f'使用设备: {device}')

    # ================= 1. 数据 =================
    # 训练集 + 测试集（dataset.py 里已经做好了增强、标准化、DataLoader 封装）
    train_loader, test_loader = get_cifar10_dataloader(args.batch_size, args.data_root)

    # 从训练集里切出 10% 当"验证集"：模型永远不会在验证集上更新参数，
    # 只用来每轮检查"学过头没有"（过拟合的探测器）
    train_dataset = train_loader.dataset
    n_val = int(len(train_dataset) * args.val_ratio)
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f'数据划分 -> 训练 {n_train} / 验证 {n_val} / 测试 {len(test_loader.dataset)}')

    # ================= 2. 模型 / 损失 / 优化器 =================
    model = create_model(args.model).to(device)      # 把模型搬到 GPU（或 CPU）
    criterion = nn.CrossEntropyLoss()                # 分类任务的标准损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam 优化器

    # ================= 3. 记录工具 =================
    run_name = f'{args.model}_{datetime.now():%m%d_%H%M%S}'
    writer = SummaryWriter(f'logs/tensorboard/{run_name}')   # TensorBoard 记录
    os.makedirs('checkpoints', exist_ok=True)

    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0.0

    # ================= 4. 训练循环（核心） =================
    for epoch in range(1, args.epochs + 1):
        # model.train()：切到训练模式（BatchNorm 会使用当前 batch 的统计量）
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start = time.time()

        # 一个 epoch = 把所有训练数据完整过一遍
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)            # 1) 前向传播：模型看图，给出 10 类得分
            loss = criterion(outputs, labels)  # 2) 算损失：预测和真实答案差多少

            optimizer.zero_grad()              # 3) 清空上一次留下的梯度（防止累加）
            loss.backward()                    # 4) 反向传播：算出每个参数该怎么调
            optimizer.step()                   # 5) 更新参数：按梯度反方向走一步（步长=学习率）

            # 顺手统计这一批的训练损失和准确率，用于打印
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # 每轮结束后，到验证集上检查一次"泛化水平"
        val_loss, val_acc = evaluate(model, val_loader, device)

        # 记录 + 画曲线
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        # 只在验证集准确率创新高时保存模型（这样 checkpoints/ 里永远是最好的模型）
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(f'checkpoints/{args.model}_best.pth',
                            model, optimizer, epoch, best_acc)

        print(f'epoch {epoch:3d}/{args.epochs} | '
              f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.1f}s')

    # ================= 5. 收尾 =================
    writer.close()
    plot_curves(train_losses, val_losses, val_accs, f'logs/{args.model}_curves.png')
    print(f'训练完成！验证集最好准确率: {best_acc:.4f}，曲线已保存到 logs/{args.model}_curves.png')

    # 用测试集评估一次最终模型（测试集只在全部训练结束后用一次，保证结果可信）
    best_path = f'checkpoints/{args.model}_best.pth'
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f'测试集结果: loss {test_loss:.4f} acc {test_acc:.4f}')


if __name__ == '__main__':
    main()
