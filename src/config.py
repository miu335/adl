import json
import warnings
from pathlib import Path

#  路径定义
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

class Config:
    def __init__(self):
        self._configs = {}
        self._config_sources = {} 
        self._load_all_configs()

    def _load_all_configs(self):
        """遍历并加载所有 JSON，冲突时保留 default.json 的值并报 Warning"""
        if not CONFIG_DIR.exists():
            raise FileNotFoundError(f"配置目录不存在: {CONFIG_DIR}")

        json_files = list(CONFIG_DIR.glob("*.json"))
        
        # 确保 default.json 排在最前面优先加载
        # 其他文件按字母顺序排序
        json_files.sort(key=lambda x: (x.name != "default.json", x.name))

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if not isinstance(data, dict):
                         raise TypeError(f"配置文件 {file_path.name} 的根节点必须是字典 (JSON Object)")

                    # 遍历当前文件的所有 key
                    for key, value in data.items():
                        if key in self._configs:
                            # 发生冲突，发出 Warning，并跳过（保留已有的值）
                            warn_msg = (
                                f"配置冲突: 键 '{key}' 在 '{file_path.name}' 和 "
                                f"'{self._config_sources[key]}' 中重复。"
                                f"已保留 '{self._config_sources[key]}' 中的值。"
                            )
                            warnings.warn(warn_msg, UserWarning)
                            continue  # 跳过赋值，保留先加载的（即 default.json 的）值                       
                        # 无冲突
                        self._configs[key] = value
                        self._config_sources[key] = file_path.name
                        
                print(f"[Info] 成功加载配置文件: {file_path.name}")
                
            except json.JSONDecodeError as e:
                print(f"[Error] 解析 JSON 失败 {file_path.name}: {e}")
            except Exception as e:
                print(f"[Error] 加载 {file_path.name} 失败: {e}")

    def get(self, key: str, default=None):
        """
        对外暴露的唯一获取方法。
        支持 a.b.c 形式的嵌套查找。
        """
        if key in self._configs:
            return self._configs[key]
        
        if '.' in key:
            keys = key.split('.')
            current = self._configs
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current

        return default

# 全局单例
cfg = Config()

# 常用配置
LOG_PATH = cfg.get("log_path")
SIM_PATH = cfg.get("sim_path")