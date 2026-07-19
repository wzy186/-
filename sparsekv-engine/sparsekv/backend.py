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

        # 内部缓存管理器（按 layer 初始化时传入）
        self._kv_manager = None

        # 回退到原生 PagedAttention 用于短序列
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
        """前向传播

        query: [num_tokens, num_heads * head_size]
        key/value: [num_tokens, num_kv_heads * head_size]
        kv_cache: vLLM 的 paged KV cache
        """
        num_tokens = query.size(0)
        seq_len = getattr(attn_metadata, "seq_lens", [num_tokens])[0]

        # 短序列回退到原生实现
        if seq_len < self.sparse_threshold or not CUDA_AVAILABLE:
            return self._fallback(
                query, key, value, kv_cache, attn_metadata
            )

        # 长序列：使用稀疏 Attention
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

        # Reshape: [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        q = query.view(num_tokens, num_heads, head_size).permute(1, 0, 2)

        # 从 kv_cache 中取出完整的 K, V（简化版，实际应从 paged cache gather）
        if kv_cache is not None:
            # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            # 这里简化处理，实际需要做 block gather
            k_cache, v_cache = kv_cache[0], kv_cache[1]
            # 取前 num_tokens 个位置
            k = k_cache[:num_tokens].permute(1, 0, 2)  # [num_kv_heads, seq, head_size]
            v = v_cache[:num_tokens].permute(1, 0, 2)
        else:
            k = key.view(num_tokens, -1, head_size).permute(1, 0, 2)
            v = value.view(num_tokens, -1, head_size).permute(1, 0, 2)

        # 处理 GQA: 重复 KV head 匹配 Q head
        if k.size(0) < num_heads:
            repeat_factor = num_heads // k.size(0)
            k = k.repeat_interleave(repeat_factor, dim=0)
            v = v.repeat_interleave(repeat_factor, dim=0)

        # Step 1: 选择 top-k KV 索引
        topk_idx = sparsekv_cuda.select_topk_kv(
            q.squeeze(1),  # [num_heads, head_size] for first token
            k,
            topk,
            self.scale
        )

        # Step 2: 稀疏 Attention 计算
        output = sparsekv_cuda.sparse_attention_forward(
            q.squeeze(1), k, v, topk_idx, self.scale
        )

        # Reshape 回 vLLM 格式: [num_tokens, num_heads * head_size]
        output = output.unsqueeze(1).permute(1, 0, 2).contiguous()
        return output.view(num_tokens, -1)

    def _init_kv_manager(self, num_layers: int):
        """延迟初始化 KV 管理器（需要知道 layer 数量）"""
        from .cache_manager import SparseKVCacheManager, FP8KVCacheManager
        # 实际项目中这里会根据 layer id 分别管理
        pass
