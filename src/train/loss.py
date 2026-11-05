# src/train/loss.py
"""按配置构造损失函数。"""

import torch.nn as nn

from src.config import LOSS
from src.utils.logger import logger


def build_loss(name=None):
    name = name or LOSS
    name = name.lower()

    if name == "mse":
        loss_fn = nn.MSELoss()
    elif name == "huber":
        loss_fn = nn.HuberLoss()
    elif name == "mae":
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f"不支持的 loss: {name}，可选: mse/huber/mae")

    logger.info(f"损失函数: {name} -> {loss_fn.__class__.__name__}")
    return loss_fn