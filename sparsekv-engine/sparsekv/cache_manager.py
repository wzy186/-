"""KV Cache 管理模块

实现 FP8 量化存储 + Top-K 稀疏化选择，降低长文本推理显存占用。
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

try:
    import sparsekv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class FP8KVCacheManager:
    """FP8 量化 KV Cache 管理器

    将 FP16/BF16 的 KV Cache 量化为 FP8 (E4M3)，显存占用减半。
    每个 tensor 单独维护 scale，支持动态范围量化。
    """

    def __init__(self, num_heads: int, head_dim: int, max_seq_len: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # FP8 存储: [num_layers, 2(K/V), num_heads, max_seq_len, head_dim]
        self.k_cache = torch.empty(
            num_heads, max_seq_len, head_dim,
            dtype=torch.int8,
            device="cuda"
        )
        self.v_cache = torch.empty(
            num_heads, max_seq_len, head_dim,
            dtype=torch.int8,
            device="cuda"
        )
        # 每层每 head 的 scale
        self.k_scale = torch.ones(num_heads, 1, device="cuda", dtype=torch.float32)
        self.v_scale = torch.ones(num_heads, 1, device="cuda", dtype=torch.float32)

    def quantize_and_store(
        self,
        k: torch.Tensor,   # [num_heads, seq_len, head_dim]
        v: torch.Tensor,   # [num_heads, seq_len, head_dim]
        start_pos: int
    ):
        """量化并写入 KV Cache"""
        num_heads, seq_len, head_dim = k.shape

        # 计算每个 head 的 scale (per-head dynamic scale)
        k_max = k.abs().amax(dim=(1, 2), keepdim=True)  # [num_heads, 1, 1]
        v_max = v.abs().amax(dim=(1, 2), keepdim=True)

        # E4M3 最大可表示值 ~448
        k_scale = k_max / 448.0
        v_scale = v_max / 448.0
        k_scale = torch.clamp(k_scale, min=1e-12)
        v_scale = torch.clamp(v_scale, min=1e-12)

        # 量化到 int8 (模拟 FP8)
        k_quant = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
        v_quant = (v / v_scale).round().clamp(-128, 127).to(torch.int8)

        # 写入 cache
        end_pos = start_pos + seq_len
        self.k_cache[:, start_pos:end_pos, :] = k_quant
        self.v_cache[:, start_pos:end_pos, :] = v_quant
        self.k_scale = k_scale.squeeze(-1)
        self.v_scale = v_scale.squeeze(-1)

    def dequantize_for_attention(
        self,
        start_pos: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """反量化指定范围的 KV，用于 Attention 计算"""
        k_quant = self.k_cache[:, start_pos:start_pos + seq_len, :]
        v_quant = self.v_cache[:, start_pos:start_pos + seq_len, :]

        # [num_heads, seq_len, head_dim] * [num_heads, 1, 1]
        k = k_quant.float() * self.k_scale.unsqueeze(1)
        v = v_quant.float() * self.v_scale.unsqueeze(1)

        return k.half(), v.half()

    def get_cache_usage_bytes(self) -> int:
        """返回当前 cache 占用的字节数"""
        return (
            self.k_cache.numel() * 1 +  # int8
            self.v_cache.numel() * 1 +
            self.k_scale.numel() * 4 +
            self.v_scale.numel() * 4
        )


class SparseKVCacheManager:
    """稀疏 KV Cache 管理器

    维护每个 head 的 Top-K 重要 KV 索引，非重要 KV 可以：
    1. 丢弃（极端压缩）
    2. 量化为更低精度（温和压缩）
    3. 转移到 CPU 内存（offload）
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        sparse_ratio: float = 0.3,
        device: str = "cuda"
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.sparse_ratio = sparse_ratio
        self.topk = max(1, int(max_seq_len * sparse_ratio))
        self.device = device

        # 全精度 KV Cache（用于对比和 topk 选择）
        self.k_cache = torch.zeros(
            num_heads, max_seq_len, head_dim,
            dtype=torch.float16, device=device
        )
        self.v_cache = torch.zeros(
            num_heads, max_seq_len, head_dim,
            dtype=torch.float16, device=device
        )

        # Top-K 索引: [num_heads, max_topk_per_chunk]
        # 为了减少重计算，每累积一定长度后更新一次 topk
        self.topk_idx = torch.zeros(
            num_heads, self.topk, dtype=torch.int32, device=device
        )
        self.cache_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """追加新的 KV，并更新 topk 索引"""
        seq_len = k.size(1)
        start = self.cache_len
        end = start + seq_len

        self.k_cache[:, start:end, :] = k
        self.v_cache[:, start:end, :] = v
        self.cache_len = end

        # 每累积 512 token 或序列结束时，重算 topk
        if end >= self.max_seq_len or end % 512 == 0:
            self._refresh_topk()

    def _refresh_topk(self):
        """基于当前 Q（用 K 近似）刷新 topk 索引"""
        if not CUDA_AVAILABLE or self.cache_len < self.topk:
            # fallback: 均匀采样
            self.topk_idx = torch.linspace(
                0, self.cache_len - 1, self.topk,
                dtype=torch.int32, device=self.device
            )[None, :].expand(self.num_heads, -1)
            return

        # 用最新的 K 作为 Query 的代理，计算重要性分数
        # 实际推理中应该用当前 step 的 Q，这里做近似
        q_proxy = self.k_cache[:, self.cache_len - 1, :]  # [num_heads, head_dim]
        scale = 1.0 / math.sqrt(self.head_dim)

        # 调用 CUDA topk 选择
        self.topk_idx = sparsekv_cuda.select_topk_kv(
            q_proxy, self.k_cache[:, :self.cache_len, :],
            self.topk, scale
        )

    def get_sparse_kv(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取稀疏化后的 K, V 和索引"""
        if self.cache_len == 0:
            return (
                torch.empty(0, self.head_dim, dtype=torch.float16, device=self.device),
                torch.empty(0, self.head_dim, dtype=torch.float16, device=self.device),
                torch.empty(0, dtype=torch.int32, device=self.device)
            )

        # 确保 topk 是最新的
        self._refresh_topk()

        # 收集 topk 位置的 KV
        k_sparse = torch.zeros(
            self.num_heads, self.topk, self.head_dim,
            dtype=torch.float16, device=self.device
        )
        v_sparse = torch.zeros_like(k_sparse)

        for h in range(self.num_heads):
            idx = self.topk_idx[h].long()
            idx = torch.clamp(idx, 0, self.cache_len - 1)
            k_sparse[h] = self.k_cache[h, idx]
            v_sparse[h] = self.v_cache[h, idx]

        return k_sparse, v_sparse, self.topk_idx

    def get_full_kv(self, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取完整 KV（用于对比测试）"""
        if seq_len is None:
            seq_len = self.cache_len
        return (
            self.k_cache[:, :seq_len, :],
            self.v_cache[:, :seq_len, :]
        )
