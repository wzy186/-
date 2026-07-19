"""LongBench 风格长文本 Benchmark

对比原版 vLLM 和 SparseKV-Engine 在长文本场景下的性能。
"""

import argparse
import time
import torch
import numpy as np

import sys
sys.path.insert(0, "/Users/didi/Desktop/sparsekv-engine")

from sparsekv.cache_manager import SparseKVCacheManager, FP8KVCacheManager
from sparsekv.utils import BenchmarkResult, reset_peak_memory, measure_peak_memory, compare_results

try:
    import sparsekv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("[WARN] sparsekv_cuda not compiled, using PyTorch fallback")
    CUDA_AVAILABLE = False


def pytorch_fallback_attention(q, k, v, scale):
    """纯 PyTorch 实现的 attention，用于无 CUDA 环境 fallback"""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def benchmark_original(seq_len: int, num_heads: int, head_dim: int, num_runs: int = 10):
    """基准测试：标准 Full Attention"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = BenchmarkResult("Original Full Attention")

    reset_peak_memory()

    for _ in range(num_runs):
        q = torch.randn(num_heads, 1, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        scale = 1.0 / (head_dim ** 0.5)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        out = pytorch_fallback_attention(q, k, v, scale)

        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        throughput = 1000.0 / latency_ms
        result.add_run(latency_ms, throughput)

    result.peak_memory_gb = measure_peak_memory()
    return result


def benchmark_sparsekv(seq_len: int, num_heads: int, head_dim: int, sparse_ratio: float, num_runs: int = 10):
    """SparseKV 测试"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = BenchmarkResult(f"SparseKV (ratio={sparse_ratio})")
    topk = max(1, int(seq_len * sparse_ratio))

    reset_peak_memory()

    for _ in range(num_runs):
        q = torch.randn(num_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        scale = 1.0 / (head_dim ** 0.5)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        if CUDA_AVAILABLE:
            topk_idx = sparsekv_cuda.select_topk_kv(q, k, topk, scale)
            out = sparsekv_cuda.sparse_attention_forward(q, k, v, topk_idx, scale)
        else:
            idx = torch.linspace(0, seq_len - 1, topk, dtype=torch.long, device=device)
            k_sparse = k[:, idx, :]
            v_sparse = v[:, idx, :]
            q_ = q.unsqueeze(1)
            out = pytorch_fallback_attention(q_, k_sparse, v_sparse, scale).squeeze(1)

        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        throughput = 1000.0 / latency_ms
        result.add_run(latency_ms, throughput)

    result.peak_memory_gb = measure_peak_memory()
    return result


def benchmark_fp8_cache(seq_len: int, num_heads: int, head_dim: int, num_runs: int = 10):
    """FP8 Cache 测试"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = BenchmarkResult("FP8 KV Cache")

    reset_peak_memory()
    cache_mgr = FP8KVCacheManager(num_heads, head_dim, seq_len * 2)

    for _ in range(num_runs):
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        start = time.perf_counter()
        cache_mgr.quantize_and_store(k, v, 0)
        k_dq, v_dq = cache_mgr.dequantize_for_attention(0, seq_len)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        throughput = seq_len / (latency_ms / 1000.0)
        result.add_run(latency_ms, throughput)

    result.peak_memory_gb = measure_peak_memory()
    fp16_bytes = num_heads * seq_len * head_dim * 2 * 2
    fp8_bytes = cache_mgr.get_cache_usage_bytes()
    saving = (1 - fp8_bytes / fp16_bytes) * 100
    print(f"  [FP8] Cache size: FP16={fp16_bytes/1024**2:.1f}MB, FP8={fp8_bytes/1024**2:.1f}MB, Saving={saving:.1f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="SparseKV Benchmark")
    parser.add_argument("--seq-len", type=int, default=32768, help="Sequence length")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--sparse-ratio", type=float, default=0.3, help="Sparse ratio")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# SparseKV Benchmark")
    print(f"# Seq Length: {args.seq_len}, Heads: {args.num_heads}, Head Dim: {args.head_dim}")
    print(f"# Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'#'*60}")

    baseline = benchmark_original(args.seq_len, args.num_heads, args.head_dim, args.num_runs)
    baseline.print_summary()

    sparse = benchmark_sparsekv(args.seq_len, args.num_heads, args.head_dim, args.sparse_ratio, args.num_runs)
    sparse.print_summary()

    fp8 = benchmark_fp8_cache(args.seq_len, args.num_heads, args.head_dim, args.num_runs)
    fp8.print_summary()

    compare_results(baseline, sparse)


if __name__ == "__main__":
    main()
