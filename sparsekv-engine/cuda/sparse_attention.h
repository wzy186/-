#pragma once

#include <torch/extension.h>

// 稀疏 Attention CUDA Kernel 声明
// Q: [num_heads, head_dim]
// K, V: [num_heads, seq_len, head_dim]
// topk_idx: [num_heads, topk] - 每个 head 保留的 KV 索引
// out: [num_heads, head_dim]
torch::Tensor sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor topk_idx,
    float scale
);

// FP8 量化辅助: 将 FP16/BF16 KV Cache 量化为 FP8
torch::Tensor quantize_fp8(torch::Tensor x, torch::Tensor scale);
torch::Tensor dequantize_fp8(torch::Tensor x, torch::Tensor scale);

// 根据 attention score 选择 top-k KV 索引
torch::Tensor select_topk_kv(
    torch::Tensor Q,
    torch::Tensor K,
    int64_t topk,
    float scale
);
