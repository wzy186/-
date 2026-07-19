"""vLLM Attention Backend 插件

实现 SparseKVAttentionBackend，作为 vLLM 的自定义 Attention 后端，
在长文本场景下自动启用稀疏 KV 模式，短文本回退到 FlashAttention。
"""

import math
from typing import Optional, List, Tuple

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention

try:
    import sparsekv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class SparseKVAttentionBackend(AttentionBackend):
    """SparseKV Attention Backend

    设计思路：
    1. 短序列 (< 4096): 直接调用原生 FlashAttention，无 overhead
    2. 长序列 (>= 4096): 启用 Top-K 稀疏 Attention + FP8 KV Cache
    3. 极长序列 (> 32K): 启用分层稀疏（Hierarchical Sparsity）
    """

    @staticmethod
    def get_name() -> str:
        return "sparsekv"

    @staticmethod
    def get_impl_cls():
        return SparseKVAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


class SparseKVAttentionImpl(torch.nn.Module):
    """SparseKV Attention 实现

    参数:
        num_heads: Query head 数量
        head_size: 每个 head 的维度
        scale: attention scale，默认 1/sqrt(head_size)
        num_kv_heads: KV head 数量（GQA/MQA 支持）
        sparse_ratio: 稀疏比例，保留多少比例的 KV
        sparse_threshold: 启用稀疏模式的序列长度阈值
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]] = None,
        sparse_ratio: float = 0.3,
        sparse_threshold: int = 4096,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale) if scale is not None else float(1.0 / math.sqrt(head_size))
        self.num_kv_heads = num_kv_heads
        self.sparse_ratio = sparse_ratio
        self.sparse_threshold = sparse_threshold
        self.alibi_slopes = alibi_slopes

        self._kv_manager = None

        self._fallback = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            scale=self.scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """前向传播"""
        num_tokens = query.size(0)
        seq_len = getattr(attn_metadata, "seq_lens", [num_tokens])[0]

        if seq_len < self.sparse_threshold or not CUDA_AVAILABLE:
            return self._fallback(
                query, key, value, kv_cache, attn_metadata
            )

        return self._sparse_attention_forward(
            query, key, value, kv_cache, attn_metadata
        )

    def _sparse_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """稀疏 Attention 核心逻辑"""
        num_tokens = query.size(0)
        num_heads = self.num_heads
        head_size = self.head_size
        topk = max(1, int(num_tokens * self.sparse_ratio))

        q = query.view(num_tokens, num_heads, head_size).permute(1, 0, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
            k = k_cache[:num_tokens].permute(1, 0, 2)
            v = v_cache[:num_tokens].permute(1, 0, 2)
        else:
            k = key.view(num_tokens, -1, head_size).permute(1, 0, 2)
            v = value.view(num_tokens, -1, head_size).permute(1, 0, 2)

        if k.size(0) < num_heads:
            repeat_factor = num_heads // k.size(0)
            k = k.repeat_interleave(repeat_factor, dim=0)
            v = v.repeat_interleave(repeat_factor, dim=0)

        topk_idx = sparsekv_cuda.select_topk_kv(
            q.squeeze(1), k, topk, self.scale
        )

        output = sparsekv_cuda.sparse_attention_forward(
            q.squeeze(1), k, v, topk_idx, self.scale
        )

        output = output.unsqueeze(1).permute(1, 0, 2).contiguous()
        return output.view(num_tokens, -1)

    def _init_kv_manager(self, num_layers: int):
        from .cache_manager import SparseKVCacheManager, FP8KVCacheManager
        pass
