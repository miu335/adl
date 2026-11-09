# src/train/trainer.py
"""训练入口：数据准备 -> 模型/损失/优化器 -> 训练循环 -> 早停 -> 保存 best.pt。"""

import os

import torch

from src.config import MODEL_TYPE, EPOCHS, MODEL_DIR, PATIENCE
from src.data.dataset import get_loaders
from src.models.base import build_model, save_model
from src.train.loss import build_loss
from src.train.optimizer import build_optimizer, build_scheduler
from src.utils.logger import logger

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X)
        n_samples += len(X)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """验证一个 epoch，返回平均 loss。"""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * len(X)
        n_samples += len(X)

    return total_loss / max(n_samples, 1)


def run_train(model_name="proxy",model_type=MODEL_TYPE):
    """
    完整训练流程:
      get_loaders -> build_model -> build_loss/optimizer/scheduler
      -> 训练循环 -> 按 val loss 保存 model_name.pt -> 早停
    """
    logger.info("===== 开始训练流程 =====")

    # 1. 数据
    train_loader, val_loader, _, data = get_loaders(model_name)

    # 2. 设备
    device = "cpu"

    # 3. 模型
    model = build_model(
        model_type,
        in_dim=len(data.feature_names),
        out_dim=len(data.target_names)
    ).to(device)

    # 4. 损失 / 优化器 / 调度器
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    # 5. 训练循环
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss = validate(model, val_loader, loss_fn, device)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | lr={current_lr:.2e}"
        )

        # 按 val loss 保存 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            logger.info(f"val loss 改善: {best_val_loss:.6f}")
            save_model(model_name, model, best_path)
        else:
            wait += 1

        # 调度器：val loss 不降则降 lr
        scheduler.step(val_loss)

        # 早停
        if wait >= PATIENCE:
            logger.info(f"早停: 连续 {PATIENCE} 个 epoch val loss 未改善")
            break

    logger.info(
        f"训练结束: best_val_loss={best_val_loss:.6f}, 权重={best_path}"
    )
    logger.info("===== 训练流程结束 =====")
    return best_path