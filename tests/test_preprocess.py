# tests/test_preprocess.py

import os
from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:   
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DIR,SIM_PATH
from src.data.preprocess import run_preprocess

RAW_DIR = os.path.dirname(SIM_PATH)  

# 与真实 CSV 同格式：分号分隔、第一行表头、第二行起数据
HEADER_LINE = (
    "width substrate in mm;width impedance line in mm;"
    "length impedance line in mm;substrate height in mm;patch height;"
    "width slot in mm;length slot in mm;patch width in mm;"
    "substrate length in mm;patch legth in mm;"
    "Frequency (GHz);S11 (dB)"
)

DATA_LINES = [
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.7999999523163;-0.058988633091261",
    "45;3.5;9.5;1.6;0.035;2.5;8.5;43;39;29.8;1.8;-0.061234567890123",
    "46;4;10;1.6;0.035;3;9;44;40;30.3;1.9;-0.070000000000000",
]


def _write_test_csv(path, data_lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER_LINE + "\n")
        for line in data_lines:
            f.write(line + "\n")


class TestPreprocessNpzMatchesCsv(unittest.TestCase):
    """npz 中的 X/y 应与写入 CSV 的模拟数据逐元素对应。"""

    @classmethod
    def setUpClass(cls):
        # 1. 在 RAW_DIR 写模拟 CSV
        cls.csv_path = os.path.join(RAW_DIR, "test_sim.csv")
        _write_test_csv(cls.csv_path, DATA_LINES)

        # 2. 调用 run_preprocess，产物落在 PROCESSED_DIR
        import src.data.preprocess as pre
        from unittest.mock import patch

        with patch.object(pre, "SIM_PATH", cls.csv_path):
            cls.out_path = pre.run_preprocess()

        # 3. 读回 npz
        cls.data = np.load(cls.out_path)

        # 4. 解析模拟数据，作为期望值（前 11 列特征，最后 1 列目标）
        cls.expected = np.array(
            [[float(v) for v in line.split(";")] for line in DATA_LINES],
            dtype=np.float32,
        )

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_npz_file_created(self):
        """npz 应生成于 PROCESSED_DIR。"""
        self.assertTrue(os.path.exists(self.out_path))
        self.assertEqual(
            os.path.dirname(os.path.abspath(self.out_path)),
            os.path.abspath(PROCESSED_DIR),
        )

    def test_shape_matches(self):
        """X 行数、列数应与模拟数据一致；y 为最后一列。"""
        X, y = self.data["X"], self.data["y"]
        self.assertEqual(X.shape, (len(DATA_LINES), self.expected.shape[1] - 1))
        self.assertEqual(y.shape, (len(DATA_LINES), 1))

    def test_X_matches_csv(self):
        """X 应与 CSV 前 11 列逐元素相等。"""
        np.testing.assert_allclose(
            self.data["X"], self.expected[:, :-1], rtol=1e-6, atol=1e-6
        )

    def test_y_matches_csv(self):
        """y 应与 CSV 最后一列逐元素相等。"""
        np.testing.assert_allclose(
            self.data["y"], self.expected[:, -1:], rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()