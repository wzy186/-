"""工具函数"""

import torch
from typing import Dict, List
import time


def measure_peak_memory():
    """测量当前 CUDA 设备的峰值显存占用"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0


def reset_peak_memory():
    """重置显存峰值计数器"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class BenchmarkResult:
    """Benchmark 结果容器"""

    def __init__(self, name: str):
        self.name = name
        self.latencies: List[float] = []
        self.throughputs: List[float] = []
        self.peak_memory_gb = 0.0

    def add_run(self, latency_ms: float, throughput_tok_per_sec: float):
        self.latencies.append(latency_ms)
        self.throughputs.append(throughput_tok_per_sec)

    def summary(self) -> Dict:
        import numpy as np
        return {
            "name": self.name,
            "latency_p50_ms": float(np.percentile(self.latencies, 50)),
            "latency_p99_ms": float(np.percentile(self.latencies, 99)),
            "throughput_mean": float(np.mean(self.throughputs)),
            "throughput_max": float(np.max(self.throughputs)),
            "peak_memory_gb": self.peak_memory_gb,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"Benchmark: {s['name']}")
        print(f"{'='*50}")
        print(f"  Latency P50:   {s['latency_p50_ms']:.2f} ms")
        print(f"  Latency P99:   {s['latency_p99_ms']:.2f} ms")
        print(f"  Throughput:    {s['throughput_mean']:.1f} tok/s (max: {s['throughput_max']:.1f})")
        print(f"  Peak Memory:   {s['peak_memory_gb']:.2f} GB")


def compare_results(baseline: BenchmarkResult, optimized: BenchmarkResult):
    """对比基准和优化结果"""
    b = baseline.summary()
    o = optimized.summary()

    print(f"\n{'='*50}")
    print("Comparison: Baseline vs SparseKV")
    print(f"{'='*50}")
    print(f"  Latency Reduction:     {(1 - o['latency_p50_ms']/b['latency_p50_ms'])*100:.1f}%")
    print(f"  Throughput Improvement: {(o['throughput_mean']/b['throughput_mean'] - 1)*100:.1f}%")
    print(f"  Memory Saving:          {(1 - o['peak_memory_gb']/b['peak_memory_gb'])*100:.1f}%")
