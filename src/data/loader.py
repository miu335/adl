# src/data/loader.py
"""仿真结果 CSV 读取与列名规范化。"""

import os
import pandas as pd

from src.utils.logger import logger

# CSV 列名（分号分隔）注意重新命名
CSV_COLUMNS = [
    "width_substrate_mm",
    "width_impedance_line_mm",
    "length_impedance_line_mm",
    "substrate_height_mm",
    "patch_height_mm",
    "width_slot_mm",
    "length_slot_mm",
    "patch_width_mm",
    "substrate_length_mm",
    "patch_length_mm",
    "frequency_ghz",
    "s11_db",
]

# 输入特征列 / 目标列
FEATURE_COLS = CSV_COLUMNS[:-1]   # 前 11 列为特征
TARGET_COLS = ["s11_db"]          # 最后一列为目标


def load_sim_csv(path):
    """
    读取天线仿真结果 CSV。

    文件特点：
      - 每列用分号 ';' 分隔
      - 第一行为表头，第二行起为数据

    返回: pandas.DataFrame，列名为 CSV_COLUMNS
    """
    if not path or not os.path.exists(path):
        logger.error(f"仿真数据文件不存在: {path}")
        raise FileNotFoundError(f"仿真数据文件不存在: {path}")

    logger.info(f"读取仿真数据: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        header=0,          
        engine="python",   
    )

    # 校验列数
    if df.shape[1] != len(CSV_COLUMNS):
        logger.error(
            f"列数不匹配: 期望 {len(CSV_COLUMNS)}，实际 {df.shape[1]}，"
            f"原始列名={list(df.columns)}"
        )
        raise ValueError(
            f"CSV 列数不匹配: 期望 {len(CSV_COLUMNS)}，实际 {df.shape[1]}"
        )

    df.columns = CSV_COLUMNS

    # 数值转换，非法值变 NaN
    for col in CSV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去掉全空行
    df = df.dropna(how="all").reset_index(drop=True)

    logger.info(f"读取完成: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def split_features_target(df):
    """拆分特征 X 与目标 y。"""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()
    return X, y