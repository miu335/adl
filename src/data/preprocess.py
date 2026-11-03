# src/data/preprocess.py
# 预处理：接收 DataFrame -> 清洗 -> 三分 -> 缩放-> 返回数组 + scaler。


from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.config import VAL_RATIO, TEST_RATIO, SEED, SCALER, TARGET_NUMBER
from src.data.loader import split_features_target
from src.utils.logger import logger


def _make_scaler(name):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none":
        return None
    raise ValueError(f"不支持的 scaler: {name}")


def run_preprocess(df, target_number=None):
   # 对已读入的 DataFrame 做清洗、三分、缩放。
 
    if target_number is None:
        target_number = TARGET_NUMBER

    logger.info("===== 开始预处理 =====")

    # 1. 划分特征/目标
    X_df, y_df = split_features_target(df, target_number)
    feature_names = list(X_df.columns)
    target_names = list(y_df.columns)

    # 2. 清洗：去掉含 NaN 的行
    mask = (X_df.notna().all(axis=1) & y_df.notna().all(axis=1)).values
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(f"删除含缺失值的行: {dropped}")
    X_df = X_df[mask].reset_index(drop=True)
    y_df = y_df[mask].reset_index(drop=True)

    X = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64)

    if len(X) < 3:
        raise ValueError(f"有效样本数过少: {len(X)}")

    # 3. 三分：先切 test，再从剩余切 val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED, shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_RATIO / (1.0 - TEST_RATIO)
    )

    # 4. 缩放：仅 train fit
    scaler_X = _make_scaler(SCALER)
    scaler_y = _make_scaler(SCALER)
    if scaler_X is not None:
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
    if scaler_y is not None:
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

    # 5. 组织返回
    result = SimpleNamespace(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        feature_names=feature_names,
        target_names=target_names,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )

    logger.info(
        f"划分完成: train={len(result.X_train)}, "
        f"val={len(result.X_val)}, test={len(result.X_test)}, "
        f"特征数={len(feature_names)}, 目标数={len(target_names)}, seed={SEED}"
    )
    logger.info("===== 预处理结束 =====")
    return result