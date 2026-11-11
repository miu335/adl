# src/train/trainer.py
"""训练器：__init__ 内部生成全部组件，fit 执行训练循环。"""

import time,os
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.config import (
    EPOCHS, PATIENCE, MODEL_DIR, MODEL_TYPE,
)
from src.models.base import build_model 
from src.data.dataset import get_loaders
from src.train.loss import build_loss
from src.train.optimizer import build_optimizer, build_scheduler
from src.utils.logger import logger


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    """
    训练器：数据准备 -> 模型/损失/优化器 -> 训练循环 -> 早停 -> 保存 pt

    用法:
        Trainer((0,-1),(-1,None),"proxy").fit()
    """
    model_name: str
    device: torch.device
    epochs: int
    patience: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader 
    data: Any
    model_type: str
    model: nn.Module
    loss_fn : Any
    optimizer: Optimizer
    scheduler: LRScheduler
    meta: dict[str, Any]

    def __init__(
        self,
        feat : tuple ,
        tgt: tuple ,
        model_name : str = f"{time.time():.2f}",
    ) -> None:
        self.model_name = model_name
        self.device =_get_device()
        self.epochs = EPOCHS
        self.patience = PATIENCE
        self.model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")

        # 1. 数据
        self.train_loader, self.val_loader, self.test_loader, self.data = get_loaders(
            feature_pos = feat, target_pos = tgt
        )
        # 2. 模型
        self.model_type = MODEL_TYPE
        self.model = build_model(
            self.model_type,
            in_dim=len(self.data.feature_names),
            out_dim=len(self.data.target_names),
        ).to(self.device)

        # 3. 损失 / 优化器 / 调度器
        self.loss_fn = build_loss()
        self.optimizer = build_optimizer(self.model)
        self.scheduler = build_scheduler(self.optimizer)

        # 4. 元信息
        self.meta = {
            "feature_names": self.data.feature_names,
            "target_names": self.data.target_names,
        }

        # 5. 训练状态
        self.best_val_loss: float = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch: int = -1
        self.wait: int = 0
        self.history: list[dict[str, Any]] = []

        logger.info(
            f"训练器初始化完成: model_path  = {self.model_path}! "
        )

    # 单 epoch
    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(X)
            n_samples += len(X)
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item() * len(X)
            n_samples += len(X)
        return total_loss / max(n_samples, 1)

    # 最优更新 / 保存
    def _update_best(self, val_loss: float, epoch: int):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            self.wait = 0
            logger.info(f"更新最优: epoch={epoch}, val_loss={val_loss:.6f}")
        else:
            self.wait += 1

    def _save_best(self):
        if self.best_state is None:
            logger.error("未产生最优权重，跳过保存")
            return None
        self.model.load_state_dict(self.best_state)
        meta = dict(self.meta)
        meta.update({
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        })
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"{self.model_name}模型权重已保存: {self.model_path}")

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def fit(self) -> str | None:
        logger.info(f"===== 开始训练: {self.model_name} =====")
        t_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            t_epoch = time.perf_counter()

            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            epoch_time = time.perf_counter() - t_epoch
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | lr={lr:.2e} | 耗时={epoch_time:.2f}s"
            )
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "time": epoch_time,
            })

            self._update_best(val_loss, epoch)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if self.wait >= self.patience:
                logger.info(f"早停: 连续 {self.patience} 个 epoch 未改善")
                break

        self._save_best()        
        total_time = time.perf_counter() - t_start
        logger.info(
            f"训练结束: best_epoch={self.best_epoch}, "
            f"best_val_loss={self.best_val_loss:.6f}, "
            f"总耗时={total_time:.2f}s ({total_time / 60:.2f}min)\n"
            "===== 训练流程结束 ====="
        )
