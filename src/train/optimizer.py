# src/train/optimizer.py
"""按配置构造优化器与学习率调度器。"""

import torch.optim as optim

from src.config import LR, WEIGHT_DECAY
from src.utils.logger import logger


def build_optimizer(model, lr=None, weight_decay=None):
    """
    构造 AdamW 优化器。
    """
    lr = LR if lr is None else lr
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    logger.info(f"优化器: AdamW, lr={lr}, weight_decay={weight_decay}")
    return optimizer


def build_scheduler(optimizer, patience=10, factor=0.5, min_lr=1e-6):
    """
    构造 ReduceLROnPlateau 调度器： loss 不降时降 lr。

    参数:
        patience: 容忍多少个 epoch 不降
        factor: 降 lr 的倍率
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )
    logger.info(
        f"调度器: ReduceLROnPlateau, patience={patience}, "
        f"factor={factor}, min_lr={min_lr}"
    )
    return scheduler