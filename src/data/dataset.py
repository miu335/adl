# src/data/dataset.py
"""从 CSV 一步生成 train/val/test 三个 DataLoader。"""

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.config import SIM_PATH, BATCH_SIZE
from src.data.loader import load_sim_csv
from src.data.preprocess import run_preprocess
from src.utils.logger import logger

_LOADER_CACHE: dict = {}

def _make_loader(X, y, batch_size, shuffle, name):
    """构造单个 DataLoader。"""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info(
        f"DataLoader[{name}]: samples={len(ds)}, "
        f"batch_size={batch_size}, shuffle={shuffle}, batches={len(loader)}"
    )
    return loader


def get_loaders(path=None, batch_size=None, feature_pos: tuple = (0, -1), target_pos: tuple = (-1, None)):
    """
    从 CSV 生成 train/val/test 三个 DataLoader。
    load_sim_csv -> run_preprocess -> 构造 DataLoader。
    """
    path = path or SIM_PATH
    batch_size = batch_size or BATCH_SIZE

    logger.info("===== 构建 DataLoader =====")

    # 1. 读 CSV
    df = load_sim_csv(path)

    # 2. 预处理（返回数组 + scaler）
    data = run_preprocess(df, feature_pos, target_pos)

    # 3. 构造三个 loader
    train_loader = _make_loader(
        data.X_train, data.y_train, batch_size, shuffle=True, name="train"
    )
    val_loader = _make_loader(
        data.X_val, data.y_val, batch_size, shuffle=False, name="val"
    )
    test_loader = _make_loader(
        data.X_test, data.y_test, batch_size, shuffle=False, name="test"
    )

    logger.info(
        f"DataLoader 构建完成: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    return train_loader, val_loader, test_loader, data