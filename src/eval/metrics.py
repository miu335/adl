# src/eval/metrics.py
"""回归评估指标。"""

import numpy as np


def rmse(y_true, y_pred):
    """均方根误差。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """平均绝对误差。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    """决定系数 R²。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def max_abs_error(y_true, y_pred):
    """最大绝对误差。"""
    return float(np.max(np.abs(y_true - y_pred)))


def compute_all(y_true, y_pred):
    """一次算全部指标，返回 dict。"""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "max_abs_error": max_abs_error(y_true, y_pred),
    }