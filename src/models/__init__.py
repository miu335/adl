# src/models/__init__.py
"""模型包：import 即注册。"""

from src.models import base
from src.models import mlp 
__all__ = ["base", "mlp"]