# src/eval/validator.py
"""评估入口：加载 model_name.pt，test 上算指标（反变换回 dB），出图。"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.ioff()  

from src.config import MODEL_TYPE, FIG_DIR, MODEL_DIR

from src.data.dataset import get_loaders
from src.eval.metrics import compute_all
from src.models.base import build_model, get_model_path
from src.utils.logger import logger


@torch.no_grad()
def _predict(model, loader, device):
    """返回 test 上的预测与真值（缩放后）。"""
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def _inverse_y(y_scaled, scaler_y):
    if scaler_y is None:
        return y_scaled
    return scaler_y.inverse_transform(y_scaled)


def _plot(y_true, y_pred, fig_dir, target_names):
    """预测 vs 真值散点图 + 误差分布图。"""
    os.makedirs(fig_dir, exist_ok=True)

    yt = y_true[:, 0]
    yp = y_pred[:, 0]
    name = target_names[0] if target_names else "target"

    # 1. 预测 vs 真值
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=8, alpha=0.6)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title("Prediction vs Truth")
    plt.tight_layout()
    scatter_path = os.path.join(fig_dir, "pred_vs_true.png")
    plt.savefig(scatter_path, dpi=120)
    plt.close()
    logger.info(f"图保存: {scatter_path}")

    # 2. 误差分布
    err = yp - yt
    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=30, alpha=0.7)
    plt.xlabel("Error (pred - true)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.tight_layout()
    hist_path = os.path.join(fig_dir, "error_dist.png")
    plt.savefig(hist_path, dpi=120)
    plt.close()
    logger.info(f"图保存: {hist_path}")


def run_evaluate(model_name="proxy"):
    """
    评估流程:
      get_loaders -> 建模型 -> 加载 ckpt -> test 预测
      -> 反变换回 dB -> 算指标 -> 出图 -> 存 JSON
    """
    logger.info("===== 开始评估流程 =====")

    ckpt = get_model_path(model_name)

    if not os.path.exists(ckpt):
        logger.error(f"权重不存在: {ckpt}")
        raise FileNotFoundError(f"权重不存在: {ckpt}")

    # 1. 数据
    _, _, test_loader, data = get_loaders(model_name)

    # 2. 设备与模型
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = build_model(
        MODEL_TYPE,
        in_dim=len(data.feature_names),
        out_dim=len(data.target_names)
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    logger.info(f"加载权重: {ckpt}")

    # 3. 预测（缩放后）
    y_pred_scaled, y_true_scaled = _predict(model, test_loader, device)

    # 4. 反变换回原始尺度（dB）
    y_pred = _inverse_y(y_pred_scaled, data.scaler_y)
    y_true = _inverse_y(y_true_scaled, data.scaler_y)

    # 5. 指标
    metrics = compute_all(y_true, y_pred)
    logger.info(
        f"Test 指标: rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}, "
        f"r2={metrics['r2']:.6f}, max_abs_error={metrics['max_abs_error']:.6f}"
    )

    # 6. 出图
    _plot(y_true, y_pred, FIG_DIR, data.target_names)

    # 7. 保存指标 JSON
    os.makedirs(FIG_DIR, exist_ok=True)
    metrics_path = os.path.join(FIG_DIR, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"指标保存: {metrics_path}")

    logger.info("===== 评估流程结束 =====")
    return metrics