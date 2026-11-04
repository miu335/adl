# src/models/base.py
"""模型注册表：按名称注册与构建模型。"""
from typing import Dict, Type
import torch.nn as nn

from src.utils.logger import logger


# 名称 -> 模型类
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name):
    """
    把模型类注册到 MODEL_REGISTRY。
    用法:
        @register_model("mlp")
        class MLP(nn.Module):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, nn.Module):
            raise TypeError(f"注册的模型必须是 nn.Module 子类: {cls}")
        if name in MODEL_REGISTRY:
            logger.warning(f"模型名重复注册，将覆盖: {name}")
        MODEL_REGISTRY[name] = cls
        logger.info(f"注册模型: {name} -> {cls.__name__}")
        return cls
    return decorator


def build_model(name, **kwargs):
    """
    按名称构建模型。
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"未注册的模型: {name}，可用模型: {sorted(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[name]
    model = cls(**kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"构建模型: {name} ({cls.__name__}), 可训练参数={n_params}")
    return model
