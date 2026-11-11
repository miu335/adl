# tests/test_dataset.py
# 生成临时 CSV，验证三个 loader 规格符合模拟文件。


import os
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.config import SIM_PATH, BATCH_SIZE
from src.data.dataset import get_loaders


HEADER_LINE = (
    "width substrate in mm;width impedance line in mm;"
    "length impedance line in mm;substrate height in mm;patch height;"
    "width slot in mm;length slot in mm;patch width in mm;"
    "substrate length in mm;patch legth in mm;"
    "Frequency (GHz);S11 (dB)"
)

DATA_LINES = [
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.7999999523163;-0.058988633091261",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8009999990463;-0.063706531642095",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8020000457764;-0.068245073382524",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8029999732971;-0.072568932249117",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8040000200272;-0.076647817143775",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8049999475479;-0.080453490410422",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8059999942780;-0.083958657758616",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8070000410080;-0.087141264066171",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8079999685287;-0.089979141416584",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8090000152588;-0.092456028062987",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8099999427795;-0.094556663196802",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8109999895096;-0.096271361085375",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8120000362396;-0.097591291058395",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8129999637604;-0.098512157641751",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8140000104904;-0.099032746387436",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8150000572205;-0.099155317262989",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8159999847412;-0.098884495725008",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8170000314713;-0.098230158993198",
    "44;3;9;1.6;0.035;2;8;42;38;29.3;1.8179999589920;-0.097201908711716",
]


def _write_test_csv(path):
    """模拟CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER_LINE + "\n")
        for line in DATA_LINES:
            f.write(line + "\n")


class TestGetLoaders(unittest.TestCase):
    """验证 get_loaders 返回的三个 loader 规格符合模拟文件。"""

    @classmethod
    def setUpClass(cls):
        # 在 SIM_PATH 同级目录写临时 CSV
        sim_dir = os.path.dirname(os.path.abspath(SIM_PATH))
        cls.csv_path = os.path.join(sim_dir, "test_dataset_sim.csv")
        _write_test_csv(cls.csv_path)

        cls.train_loader, cls.val_loader, cls.test_loader, _ = get_loaders(
        path=cls.csv_path,feature_pos=(0, -1), target_pos=(-1, None)
        )

        sample_x, sample_y = cls.train_loader.dataset[0]
        cls.n_total = len(DATA_LINES)              # 19
        cls.n_feat = sample_x.shape[0]   # 11
        cls.n_tgt = sample_y.shape[0]     # 1

    @classmethod
    def tearDownClass(cls):
        """测试结束后删除模拟 CSV。"""
        if os.path.exists(cls.csv_path):
            os.remove(cls.csv_path)
            
    def test_total_samples(self):
        """三分样本数之和 = 模拟文件行数。"""
        total = (
            len(self.train_loader.dataset)
            + len(self.val_loader.dataset)
            + len(self.test_loader.dataset)
        )
        self.assertEqual(total, self.n_total)

    def test_feature_target_dims(self):
        """特征 11 维，目标 1 维。"""
        self.assertEqual(self.n_feat, 11)
        self.assertEqual(self.n_tgt, 1)


if __name__ == "__main__":
    unittest.main()