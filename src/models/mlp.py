# src/models/mlp.py
""" MLP 回归模型 """

import torch.nn as nn

from src.models.base import register_model

@register_model("mlp")
class MLP(nn.Module):
    """
    多层感知机回归模型。

    结构: Linear -> Activation -> Dropout -> ... -> Linear(out_dim)

    参数:
        in_dim: 输入特征维度（= len(feature_names)）
        out_dim: 输出维度（= len(target_names)）
        hidden: 隐藏层宽度列表，如 [256, 256, 128]
        dropout: dropout 概率，0 表示不启用
        activation: 激活函数名
    """

    def __init__(self, in_dim, out_dim, hidden=(256, 256, 128), dropout=0.1):       
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim/out_dim 必须 > 0: {in_dim}/{out_dim}")
        
        layers = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)