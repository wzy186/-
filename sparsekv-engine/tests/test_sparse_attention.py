"""Sparse Attention 单元测试"""

import sys
sys.path.insert(0, "/Users/didi/Desktop/sparsekv-engine")

import torch
import pytest

try:
    import sparsekv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def test_select_topk_shape():
    """测试 topk 索引输出形状正确"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")

    num_heads, seq_len, head_dim = 8, 1024, 64
    q = torch.randn(num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    topk = 128
    scale = 1.0 / (head_dim ** 0.5)

    idx = sparsekv_cuda.select_topk_kv(q, k, topk, scale)
    assert idx.shape == (num_heads, topk)
    assert idx.dtype == torch.int32


def test_sparse_attention_shape():
    """测试稀疏 attention 输出形状正确"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")

    num_heads, seq_len, head_dim = 8, 1024, 64
    topk = 128
    q = torch.randn(num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    topk_idx = torch.randint(0, seq_len, (num_heads, topk), dtype=torch.int32, device="cuda")
    scale = 1.0 / (head_dim ** 0.5)

    out = sparsekv_cuda.sparse_attention_forward(q, k, v, topk_idx, scale)
    assert out.shape == (num_heads, head_dim)
    assert out.dtype == torch.float16


def test_sparse_attention_numerical():
    """数值正确性：稀疏 attention 结果应与全 attention 接近（保留全部 KV 时）"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")

    num_heads, seq_len, head_dim = 4, 256, 64
    q = torch.randn(num_heads, head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
    scale = 1.0 / (head_dim ** 0.5)

    # Ground truth: full attention
    scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = torch.matmul(attn, v).squeeze(1)

    # Sparse: 保留全部 KV (topk = seq_len)
    topk_idx = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0).expand(num_heads, -1)
    q_h = q.half()
    k_h = k.half()
    v_h = v.half()
    out = sparsekv_cuda.sparse_attention_forward(q_h, k_h, v_h, topk_idx, scale)

    # 允许一定误差 (FP16 vs FP32)
    assert torch.allclose(out.float(), expected, atol=1e-2, rtol=1e-2)


def test_fp8_quantization_roundtrip():
    """测试 FP8 量化-反量化可逆性"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")

    x = torch.randn(1024, dtype=torch.float16, device="cuda")
    scale = (x.abs().max() / 448.0).unsqueeze(0)

    q = sparsekv_cuda.quantize_fp8(x, scale)
    x_dq = sparsekv_cuda.dequantize_fp8(q, scale)

    # 量化误差应小于 1%
    rel_err = (x_dq - x).abs().mean() / x.abs().mean()
    assert rel_err < 0.01


def test_cache_manager():
    """测试 FP8KVCacheManager 基本功能"""
    from sparsekv.cache_manager import FP8KVCacheManager

    mgr = FP8KVCacheManager(num_heads=8, head_dim=64, max_seq_len=1024)
    k = torch.randn(8, 512, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(8, 512, 64, dtype=torch.float16, device="cuda")

    mgr.quantize_and_store(k, v, 0)
    k_dq, v_dq = mgr.dequantize_for_attention(0, 512)

    assert k_dq.shape == k.shape
    assert v_dq.shape == v.shape
    # 允许量化误差
    assert torch.allclose(k_dq, k, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    test_select_topk_shape()
    test_sparse_attention_shape()
    test_sparse_attention_numerical()
    test_fp8_quantization_roundtrip()
    test_cache_manager()
    print("All tests passed!")
