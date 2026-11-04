# tests/test_model.py
"""模型测试：注册表 + MLP 前向形状。"""

import unittest

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch

#import src.models  # 触发注册
from src.models.base import  build_model

class TestMLP(unittest.TestCase):
    def test_forward_shape(self):
        model = build_model(
            "mlp", in_dim=11, out_dim=1, hidden=[64, 32], dropout=0.0
        )
        x = torch.randn(8, 11)
        self.assertEqual(model(x).shape, (8, 1))

    def test_multi_output(self):
        model = build_model(
            "mlp", in_dim=11, out_dim=3, hidden=[32], dropout=0.0
        )
        x = torch.randn(5, 11)
        self.assertEqual(model(x).shape, (5, 3))

if __name__ == "__main__":
    unittest.main()