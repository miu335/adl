# tests/test_eval.py
"""metrics测试。"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

from src.config import FIG_DIR
from src.eval.metrics import compute_all, mae, r2, rmse


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        y = np.array([[1.0], [2.0], [3.0]])
        m = compute_all(y, y)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["mae"], 0.0)
        self.assertAlmostEqual(m["r2"], 1.0)

    def test_rmse(self):
        y_true = np.array([[0.0], [0.0]])
        y_pred = np.array([[1.0], [-1.0]])
        self.assertAlmostEqual(rmse(y_true, y_pred), 1.0)

    def test_mae(self):
        y_true = np.array([[0.0], [0.0]])
        y_pred = np.array([[1.0], [-1.0]])
        self.assertAlmostEqual(mae(y_true, y_pred), 1.0)

    def test_r2_constant(self):
        y_true = np.array([[1.0], [1.0]])
        y_pred = np.array([[1.0], [1.0]])
        self.assertAlmostEqual(r2(y_true, y_pred), 1.0)


if __name__ == "__main__":
    unittest.main()