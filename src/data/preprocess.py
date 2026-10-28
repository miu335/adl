# src/data/preprocess.py
"""预处理入口：读取 CSV -> 清洗 -> 落盘 processed npz。"""

import os
import numpy as np

from src.config import SIM_PATH, PROCESSED_DIR
from src.data.loader import load_sim_csv, split_features_target
from src.utils.logger import logger


def run_preprocess():
    """
    读取仿真 CSV，做基础清洗后保存为 npz。
    """
    logger.info("===== 开始预处理 =====")

    df = load_sim_csv(SIM_PATH)
    X, y = split_features_target(df)

    # 清洗：删除含 NaN 的行 
    mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(f"删除含缺失值的行: {dropped}")
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # 落盘
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "raw_clean.npz")
    np.savez(
        out_path,
        X=X.to_numpy(dtype=np.float32),
        y=y.to_numpy(dtype=np.float32),
        feature_names=np.array(X.columns.tolist()),
        target_names=np.array(y.columns.tolist()),
    )
    logger.info(f"预处理完成，保存至: {out_path}，样本数={len(X)}")
    logger.info("===== 预处理结束 =====")
    return out_path