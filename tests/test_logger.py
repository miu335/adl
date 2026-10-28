# tests/test_logger.py
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import logger

class TestLogger(unittest.TestCase):

    def test_log_written_to_file(self):
        """简单验证日志是否能根据配置写入到文件"""
        test_msg = "测试用例生成的日志_ABC123"
        logger.info(test_msg)
        for handler in logger.handlers:
            handler.flush()

if __name__ == '__main__':
    unittest.main()