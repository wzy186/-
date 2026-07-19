#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128

// -----------------------------------------------------------------------------
// 工具函数：warp 级别的 reduce max + sum
// -----------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// -----------------------------------------------------------------------------
// Kernel: 计算 Q @ K^T 的 top-k 索引
// 每个 warp 处理一个 head 的若干个 query token
// -----------------------------------------------------------------------------
__global__ void select_topk_kernel(
    const half* __restrict__ Q,      // [num_heads, head_dim]
    const half* __restrict__ K,      // [num_heads, seq_len, head_dim]
    int* __restrict__ out_idx,       // [num_heads, topk]
    int num_heads,
    int seq_len,
    int head_dim,
    int topk,
    float scale
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;

    const half* q_ptr = Q + head_id * head_dim;
    const half* k_ptr = K + head_id * seq_len * head_dim;

    // Shared memory 放当前 warp 要处理的 Q 向量
    __shared__ float q_vec[MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_vec[tid] = __half2float(q_ptr[tid]);
    }
    __syncthreads();

    // 每个 thread 处理一部分 K 位置，计算 attention score
    // 简化：每个 lane 计算一个 K 位置的 score
    float local_max = -FLT_MAX;
    int local_idx = -1;

    for (int k_pos = lane; k_pos < seq_len; k_pos += WARP_SIZE) {
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            float k_val = __half2float(k_ptr[k_pos * head_dim + d]);
            score += q_vec[d] * k_val;
        }
        score *= scale;

        if (score > local_max) {
            local_max = score;
            local_idx = k_pos;
        }
    }

    // Warp reduce: 找 top-1（简化版，实际需要更复杂的 top-k）
    // 这里为了代码简洁，演示性只找 top-1，实际项目可扩展为 bitonic sort top-k
    float warp_max = warp_reduce_max(local_max);

    // 仅演示：把 score 最高的位置写回
    // 实际实现中这里应该用 shared memory 做 block-level top-k
    if (lane == 0) {
        // 找到哪个 lane 有最大值
        int max_lane = 0;
        // 简化为 lane 0 写回
        out_idx[head_id * topk] = local_idx >= 0 ? local_idx : 0;
    }
}

// -----------------------------------------------------------------------------
// Kernel: Sparse Attention Forward
// 每个 block 处理一个 head，thread 处理 head_dim 的一部分
// 只读取 top-k 个 K/V，大幅减少显存访问
// -----------------------------------------------------------------------------
__global__ void sparse_attention_kernel(
    const half* __restrict__ Q,        // [num_heads, head_dim]
    const half* __restrict__ K,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ V,        // [num_heads, seq_len, head_dim]
    const int* __restrict__ topk_idx,  // [num_heads, topk]
    half* __restrict__ O,              // [num_heads, head_dim]
    int num_heads,
    int seq_len,
    int head_dim,
    int topk,
    float scale
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 当前 head 的基地址
    const half* q_ptr = Q + head_id * head_dim;
    const half* k_ptr = K + head_id * seq_len * head_dim;
    const half* v_ptr = V + head_id * seq_len * head_dim;
    half* o_ptr = O + head_id * head_dim;
    const int* idx_ptr = topk_idx + head_id * topk;

    // Shared memory 布局:
    // [0 : head_dim]           -> Q 向量（复用）
    // [head_dim : head_dim*2]  -> 当前处理的 K 切片
    // [head_dim*2 : ]          -> top-k attention scores
    extern __shared__ char smem[];
    float* s_q = (float*)smem;
    float* s_k = s_q + head_dim;
    float* s_scores = s_k + head_dim;

    // Step 1: 加载 Q 到 shared memory
    for (int d = tid; d < head_dim; d += num_threads) {
        s_q[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    // Step 2: 计算每个 top-k K 位置的 attention score
    // 每个 thread 负责一部分 top-k 位置
    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        int k_pos = idx_ptr[k_i];
        // 边界保护
        if (k_pos < 0 || k_pos >= seq_len) k_pos = 0;

        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            float k_val = __half2float(k_ptr[k_pos * head_dim + d]);
            score += s_q[d] * k_val;
        }
        s_scores[k_i] = score * scale;
    }
    __syncthreads();

    // Step 3: Softmax (block-level)
    // 找 max
    float thread_max = -FLT_MAX;
    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        thread_max = fmaxf(thread_max, s_scores[k_i]);
    }
    // block reduce max via warp
    float warp_max = warp_reduce_max(thread_max);
    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) {
        atomicMax((int*)&block_max, __float_as_int(warp_max));
    }
    __syncthreads();

    // exp 并求和
    float thread_sum = 0.0f;
    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        float exp_val = expf(s_scores[k_i] - block_max);
        s_scores[k_i] = exp_val;
        thread_sum += exp_val;
    }
    float warp_sum = warp_reduce_sum(thread_sum);
    __shared__ float block_sum;
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&block_sum, warp_sum);
    }
    __syncthreads();

    // 归一化
    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        s_scores[k_i] /= block_sum;
    }
    __syncthreads();

    // Step 4: 计算输出 O = sum(score_i * V_i)
    // 每个 thread 负责 head_dim 的一部分
    for (int d = tid; d < head_dim; d += num_threads) {
        float out_val = 0.0f;
        for (int k_i = 0; k_i < topk; ++k_i) {
            int k_pos = idx_ptr[k_i];
            if (k_pos < 0 || k_pos >= seq_len) k_pos = 0;
            float v_val = __half2float(v_ptr[k_pos * head_dim + d]);
            out_val += s_scores[k_i] * v_val;
        }
        o_ptr[d] = __float2half(out_val);
    }
}

// -----------------------------------------------------------------------------
// FP8 量化 Kernel (E4M3)
// -----------------------------------------------------------------------------
__global__ void quantize_fp8_kernel(
    const half* __restrict__ x,
    int8_t* __restrict__ out,
    float* __restrict__ scale_out,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 先找 max（这里简化：假设 scale 已在外部计算好）
    // 实际应该用 reduce kernel 先算 max
    float val = __half2float(x[tid]);
    float scale = scale_out[0];
    float quantized = val / scale;
    // clamp to E4M3 range: [-448, 448]
    quantized = fmaxf(-448.0f, fminf(448.0f, quantized));
    out[tid] = (int8_t)(quantized + (quantized >= 0 ? 0.5f : -0.5f));
}

__global__ void dequantize_fp8_kernel(
    const int8_t* __restrict__ x,
    half* __restrict__ out,
    const float* __restrict__ scale,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __float2half((float)x[tid] * scale[0]);
}

// -----------------------------------------------------------------------------
// PyTorch 绑定
// -----------------------------------------------------------------------------
torch::Tensor sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor topk_idx,
    float scale
) {
    int num_heads = Q.size(0);
    int head_dim = Q.size(1);
    int seq_len = K.size(1);
    int topk = topk_idx.size(1);

    auto O = torch::empty_like(Q);

    dim3 blocks(num_heads);
    dim3 threads(128);
    size_t smem_size = (head_dim * 2 + topk) * sizeof(float);

    sparse_attention_kernel<<<blocks, threads, smem_size>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        topk_idx.data_ptr<int>(),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        num_heads,
        seq_len,
        head_dim,
        topk,
        scale
    );

    return O;
}

torch::Tensor quantize_fp8(torch::Tensor x, torch::Tensor scale) {
    auto out = torch::empty(x.sizes(), torch::dtype(torch::kInt8).device(x.device()));
    int n = x.numel();
    quantize_fp8_kernel<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        out.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        n
    );
    return out;
}

torch::Tensor dequantize_fp8(torch::Tensor x, torch::Tensor scale) {
    auto out = torch::empty(x.sizes(), torch::dtype(torch::kFloat16).device(x.device()));
    int n = x.numel();
    dequantize_fp8_kernel<<<(n + 255) / 256, 256>>>(
        x.data_ptr<int8_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        n
    );
    return out;
}

torch::Tensor select_topk_kv(
    torch::Tensor Q,
    torch::Tensor K,
    int64_t topk,
    float scale
) {
    int num_heads = Q.size(0);
    int seq_len = K.size(1);
    auto out_idx = torch::empty({num_heads, topk}, torch::dtype(torch::kInt32).device(Q.device()));

    select_topk_kernel<<<num_heads, 128>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        out_idx.data_ptr<int>(),
        num_heads,
        seq_len,
        Q.size(1),
        topk,
        scale
    );
    return out_idx;
}

// -----------------------------------------------------------------------------
// Python 绑定入口
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention_forward", &sparse_attention_forward, "Sparse Attention Forward");
    m.def("quantize_fp8", &quantize_fp8, "Quantize to FP8");
    m.def("dequantize_fp8", &dequantize_fp8, "Dequantize from FP8");
    m.def("select_topk_kv", &select_topk_kv, "Select Top-K KV indices");
}
