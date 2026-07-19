"""Sparse Attention vs FlashAttention 对比测试

对比维度：
1. 数值精度：同一组输入下，输出差异（MSE / Max Error）
2. 推理速度：不同序列长度下的 latency
3. 显存占用：peak memory

需要安装 FlashAttention:
    pip install flash-attn --no-build-isolation

如果安装失败，脚本会自动 fallback 到 PyTorch SDPA。
"""

import sys
sys.path.insert(0, "/Users/didi/Desktop/sparsekv-engine")

import argparse
import time
import torch
import torch.nn.functional as F

from sparsekv.utils import BenchmarkResult, reset_peak_memory, measure_peak_memory

try:
    import sparsekv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("[WARN] sparsekv_cuda not compiled, CUDA tests skipped")

FLASH_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
    print("[INFO] FlashAttention 已加载")
except ImportError:
    print("[WARN] flash-attn 未安装，使用 PyTorch SDPA 作为 baseline")


def pytorch_sdpa_attention(q, k, v, causal=True):
    """PyTorch 原生 scaled_dot_product_attention"""
    # q,k,v: [bsz, num_heads, seq_len, head_dim]
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash_attention(q, k, v, causal=True):
    """FlashAttention (或 fallback 到 SDPA)"""
    if FLASH_AVAILABLE:
        # flash_attn 期望输入 [bsz, seq_len, num_heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return flash_attn_func(q, k, v, causal=causal).transpose(1, 2)
    else:
        return pytorch_sdpa_attention(q, k, v, causal)


def sparse_attention_cuda(q, k, v, topk_ratio=0.3):
    """我们的 Sparse Attention CUDA 实现"""
    bsz, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim ** 0.5)
    topk = max(1, int(seq_len * topk_ratio))

    # 取最后一个 query（模拟 decode 阶段）
    q_last = q[:, :, -1, :]  # [bsz, heads, head_dim]

    # 展平 batch + head 维度给 CUDA kernel
    q_flat = q_last.reshape(-1, head_dim)
    k_flat = k.reshape(-1, seq_len, head_dim)
    v_flat = v.reshape(-1, seq_len, head_dim)

    topk_idx = sparsekv_cuda.select_topk_kv(q_flat, k_flat, topk, scale)
    out_flat = sparsekv_cuda.sparse_attention_forward(q_flat, k_flat, v_flat, topk_idx, scale)

    out = out_flat.view(bsz, num_heads, head_dim)
    # 扩展回 seq_len 维度（为了和 baseline 同形状对比，实际 decode 只有一个 token）
    return out.unsqueeze(2)


def benchmark_speed(seq_len: int, num_heads: int, head_dim: int, bsz: int = 1, num_runs: int = 20):
    """速度对比"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Speed Benchmark | seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
    print(f"{'='*60}")

    # Baseline: FlashAttention / SDPA
    baseline = BenchmarkResult("FlashAttention / SDPA")
    for _ in range(num_runs):
        q = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = flash_attention(q, k, v, causal=True)
        if device == "cuda":
            torch.cuda.synchronize()
        baseline.add_run((time.perf_counter() - t0) * 1000, 0)
    baseline.print_summary()

    # SparseKV
    if CUDA_AVAILABLE:
        sparse = BenchmarkResult("SparseKV (CUDA)")
        for _ in range(num_runs):
            q = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
            k = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
            v = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = sparse_attention_cuda(q, k, v, topk_ratio=0.3)
            if device == "cuda":
                torch.cuda.synchronize()
            sparse.add_run((time.perf_counter() - t0) * 1000, 0)
        sparse.print_summary()

        b = baseline.summary()
        s = sparse.summary()
        print(f"\n  Speedup: {b['latency_p50_ms'] / s['latency_p50_ms']:.2f}x")
    else:
        print("  [SKIP] CUDA not available")


def benchmark_accuracy(seq_len: int, num_heads: int, head_dim: int, bsz: int = 2):
    """数值精度对比"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Accuracy Benchmark | seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
    print(f"{'='*60}")

    q = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(bsz, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    # Baseline
    out_base = flash_attention(q, k, v, causal=False)

    # Sparse
    if CUDA_AVAILABLE:
        out_sparse = sparse_attention_cuda(q, k, v, topk_ratio=0.3)

        # 由于 sparse_attention_cuda 返回 [bsz, heads, 1, dim]，需要扩展对比
        out_sparse_exp = out_sparse.expand(-1, -1, seq_len, -1)

        mse = ((out_base.float() - out_sparse_exp.float()) ** 2).mean().item()
        max_err = (out_base.float() - out_sparse_exp.float()).abs().max().item()
        cos_sim = F.cosine_similarity(out_base.float().reshape(1, -1),
                                      out_sparse_exp.float().reshape(1, -1)).item()

        print(f"  MSE:        {mse:.6f}")
        print(f"  Max Error:  {max_err:.6f}")
        print(f"  Cosine Sim: {cos_sim:.6f}")

        assert mse < 0.1, f"MSE {mse} too high"
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim} too low"
        print("  [PASS] 精度对比通过")
    else:
        print("  [SKIP] CUDA not available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=20)
    args = parser.parse_args()

    benchmark_accuracy(args.seq_len, args.num_heads, args.head_dim, args.bsz)
    benchmark_speed(args.seq_len, args.num_heads, args.head_dim, args.bsz, args.num_runs)


if __name__ == "__main__":
    main()
