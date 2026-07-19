"""SparseKV-Engine: 长文本推理 KV Cache 压缩加速系统

基于 vLLM 的自定义 Attention 后端，实现 Top-K 稀疏 Attention + FP8 KV Cache 量化，
支持 2~4 倍长文本推理加速，显存占用降低 50%+。
"""

__version__ = "0.1.0"

from .cache_manager import FP8KVCacheManager, SparseKVCacheManager
from .backend import SparseKVAttentionBackend

__all__ = [
    "FP8KVCacheManager",
    "SparseKVCacheManager", 
    "SparseKVAttentionBackend",
]
