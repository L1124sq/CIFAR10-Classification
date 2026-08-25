# -*- coding: utf-8 -*-
"""
train.py —— 训练主程序

训练循环的核心仍然是：
    前向传播 -> 算损失 -> 反向传播 -> 更新参数

本版本在原来基础上增加了：
  1. 实验 tag：不同实验的输出不会互相覆盖
  2. ResNet18/34/50/101/152 骨干网络选择
  3. weight decay、dropout、label smoothing、cutout 等正则化选项
  4. 可选梯度累积和早停
  5. 自动记录参数数量、每轮指标、CSV/JSON 结果
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import get_cifar10_dataloader
from model import create_model
from utils import (
    append_csv,
    count_parameters,
    evaluate,
    get_device,
    plot_curves,
    save_checkpoint,
    save_json,
    set_seed,
)


MODEL_CHOICES = [
    "cnn",
    "resnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 分类训练")

    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=MODEL_CHOICES,
        help="模型名称，resnet 等价于 resnet18",
    )
    parser.add_argument("--tag", type=str, default=None, help="实验标识，用于输出文件名")
    parser.add_argument("--epochs", type=int, default=30, help="训练多少轮")
    parser.add_argument("--batch_size", type=int, default=128, help="每批多少张图")
    parser.add_argument("--accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练（AMP）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--data_root", type=str, default="./data", help="数据目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 正则化系数")
    parser.add_argument("--dropout", type=float, default=0.0, help="分类层前 Dropout 概率")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑系数")
    parser.add_argument("--use_cutout", action="store_true", help="训练集使用 RandomErasing/Cutout")

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "plateau"],
        help="学习率调度器",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="验证集连续多少轮不提升就早停，0 表示关闭",
    )

    return parser.parse_args()


def build_paths(args):
    tag = args.tag or f"{args.model}_{datetime.now():%m%d_%H%M%S}"
    checkpoint_path = f"checkpoints/{tag}_best.pth"
    curve_path = f"results/curves/{tag}_curves.png"
    tensorboard_dir = f"logs/tensorboard/{tag}"
    metrics_json_path = f"results/metrics/{tag}.json"
    return tag, checkpoint_path, curve_path, tensorboard_dir, metrics_json_path


def make_optimizer(model, args):
    # AdamW 的 weight_decay 与参数更新解耦，比普通 Adam 的 L2 实现更规范。
    # weight_decay=0 时，行为与当前基线使用的 Adam 基本一致。
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def make_scheduler(optimizer, args):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return None


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    # ================= 1. 数据 =================
    train_loader, test_loader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        use_cutout=args.use_cutout,
    )

    train_dataset = train_loader.dataset
    n_val = int(len(train_dataset) * args.val_ratio)
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    print(f"数据划分 -> 训练 {n_train} / 验证 {n_val} / 测试 {len(test_loader.dataset)}")

    # ================= 2. 模型 / 损失 / 优化器 =================
    model = create_model(args.model, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)

    total_params, trainable_params = count_parameters(model)
    print(f"模型 {args.model}: 总参数 {total_params:,}，可训练参数 {trainable_params:,}")

    use_amp = args.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        print("已启用混合精度训练（AMP）。")

    # ================= 3. 输出目录 =================
    tag, checkpoint_path, curve_path, tensorboard_dir, metrics_json_path = build_paths(args)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/curves", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    # ================= 4. 训练循环 =================
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        optimizer.zero_grad()
        last_batch_idx = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            last_batch_idx = batch_idx
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels) / args.accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            # loss 被 accum_steps 缩放，这里恢复成原始平均损失
            batch_loss = loss.item() * args.accum_steps
            running_loss += batch_loss * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # 处理最后一个不足 accum_steps 的尾部 batch
        if (last_batch_idx + 1) % args.accum_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            label_smoothing=args.label_smoothing,
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("gap", train_acc - val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_acc,
                extra={
                    "tag": tag,
                    "model": args.model,
                    "args": vars(args),
                },
            )
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"gap {train_acc - val_acc:.4f} | {time.time() - start:.1f}s"
        )

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"验证集连续 {epochs_no_improve} 轮没有提升，提前停止。")
            break

    writer.close()
    plot_curves(
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        curve_path,
    )
    print(f"训练完成！验证集最好准确率: {best_acc:.4f}，曲线已保存到 {curve_path}")

    # ================= 5. 测试集最终评估 =================
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device,
        label_smoothing=args.label_smoothing,
    )

    best_idx = val_accs.index(max(val_accs)) if val_accs else 0
    best_train_acc = train_accs[best_idx] if train_accs else 0.0
    best_gap = best_train_acc - best_acc
    final_gap = train_accs[-1] - val_accs[-1] if train_accs else 0.0

    metrics = {
        "tag": tag,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "use_amp": args.use_amp,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "use_cutout": args.use_cutout,
        "scheduler": args.scheduler,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "train_acc_at_best_epoch": best_train_acc,
        "gap_at_best_epoch": best_gap,
        "final_train_acc": train_accs[-1] if train_accs else 0.0,
        "final_val_acc": val_accs[-1] if val_accs else 0.0,
        "final_gap": final_gap,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    save_json(metrics_json_path, metrics)
    append_csv(
        "results/metrics.csv",
        metrics,
        fieldnames=list(metrics.keys()),
    )
    print(f"结果已保存到 {metrics_json_path} 和 results/metrics.csv")
    print(f"测试集结果: loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
