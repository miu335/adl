import os
import logging

from src.config import LOG_PATH


def _get_logger(name="src_logger", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 Handler（路径来自 config.config.LOG_PATH）
    log_path = LOG_PATH
    log_dir = os.path.dirname(log_path)
    print(f"[Info] 日志文件路径: {log_path}")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"无法创建日志文件 {LOG_PATH}: {e}")

    return logger


logger = _get_logger()