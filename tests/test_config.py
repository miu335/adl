# tests/config.py
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import cfg

class TestConfigSimple(unittest.TestCase):

    def test_get_top_level_key(self):
        """测试获取普通key"""
        val = cfg.get("batch_size")
        print(f"\n[测试] 获取 batch_size: {val}")
        self.assertIsNotNone(val, "配置里缺少 batch_size")

    def test_get_nested_key(self):
        """测试获取嵌套 key"""
        val = cfg.get("model.name")
        print(f"[测试] 获取 model.name: {val}")
        self.assertIsNotNone(val, "配置里缺少 model.name")

    def test_default_value(self):
        """测试获取不存在的 key 时，返回默认值"""
        val = cfg.get("i_do_not_exist", default="fallback")
        print(f"[测试] 获取不存在的 key: {val}")
        self.assertEqual(val, "fallback")

if __name__ == '__main__':
    unittest.main(verbosity=2)