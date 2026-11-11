# src/data/loader.py
# 仿真结果 CSV 读取，按 target_number 划分特征/目标。

import pandas as pd

from src.config import TARGET_NUMBER
from src.utils.logger import logger

def load_sim_csv(path):
    # 读取天线仿真结果 CSV
    logger.info(f"读取仿真数据: {path}")

    try:
        df = pd.read_csv(path, sep=";", header=0, engine="python")
        logger.info(f"成功读取数据集！当前数据维度: {df.shape} ")
    except FileNotFoundError:
        logger.error(f"未找到数据集文件: {path}")
        raise FileNotFoundError

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)

    logger.info(
        f"读取完成: {df.shape[0]} 行 × {df.shape[1]} 列，"
        f"列名: {list(df.columns)}"
    )
    return df


def split_features_target(df, feature_pos: tuple, target_pos: tuple):
    """
    返回: (X, y) 两个 DataFrame，保留原始列名
    """

    feature_cols = list(df.columns[slice(*feature_pos)])
    target_cols = list(df.columns[slice(*target_pos)])

    logger.info(
        f"划分: feature_number={len(feature_cols)},target_number={len(target_cols)} "
        f"特征({len(feature_cols)})={feature_cols}, "
        f"目标({len(target_cols)})={target_cols}"
    )

    return df[feature_cols].copy(), df[target_cols].copy()