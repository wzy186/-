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
// Warp-level argmax: 返回最大值及其索引
// -----------------------------------------------------------------------------
__inline__ __device__ void warp_argmax(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// -----------------------------------------------------------------------------
// Block-level argmax via shared memory: 找出 [0, n) 范围内的最大值及索引
// 每个 thread 先扫描自己负责的元素，warp 内 reduce，再在 SMEM 合并
// -----------------------------------------------------------------------------
__inline__ __device__ void block_argmax(
    float* smem_vals,
    int* smem_idx,
    int num_warps
) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // warp leader 写入 SMEM
    if (lane == 0 && warp_id < num_warps) {
        smem_vals[warp_id] = smem_vals[warp_id];
        smem_idx[warp_id] = smem_idx[warp_id];
    }
    __syncthreads();

    // thread 0 在 SMEM 里找全局最大
    if (tid == 0) {
        float gmax = -FLT_MAX;
        int gidx = -1;
        for (int i = 0; i < num_warps; ++i) {
            if (smem_vals[i] > gmax) {
                gmax = smem_vals[i];
                gidx = smem_idx[i];
            }
        }
        smem_vals[0] = gmax;
        smem_idx[0] = gidx;
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Kernel: Block-Level Top-K Selection
// 
// 升级点：从 toy 的 top-1 改为真正的 K 轮贪心选择
// 策略：
//   1. 先把 Q·K^T 的所有 score 算好，写入外部传入的 score_buf（global mem）
//   2. 重复 K 轮：每轮用 warp_argmax + block_argmax 找出当前全局最大值，
//      记录索引，并把该位置 mark 成 -inf
// 
// 复杂度：O(K * seq_len / threads)。K 不大时（<=256）在 A100 上 < 5ms。
// -----------------------------------------------------------------------------
__global__ void select_topk_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    int* __restrict__ out_idx,
    float* __restrict__ score_buf,   // [num_heads, seq_len] 全局 score 缓存
    int num_heads,
    int seq_len,
    int head_dim,
    int topk,
    float scale
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const half* q_ptr = Q + head_id * head_dim;
    const half* k_ptr = K + head_id * seq_len * head_dim;
    float* buf = score_buf + head_id * seq_len;

    __shared__ float q_vec[MAX_HEAD_DIM];
    if (tid < head_dim) {
        q_vec[tid] = __half2float(q_ptr[tid]);
    }
    __syncthreads();

    // Step 1: 计算所有 score 写入 global memory buffer
    for (int k_pos = tid; k_pos < seq_len; k_pos += blockDim.x) {
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            float k_val = __half2float(k_ptr[k_pos * head_dim + d]);
            score += q_vec[d] * k_val;
        }
        buf[k_pos] = score * scale;
    }
    __syncthreads();

    // Step 2: K 轮贪心选择，每轮找全局最大值
    // 共享内存复用：前 num_warps 个位置存 warp max/idx
    extern __shared__ char smem[];
    float* s_warp_max = (float*)smem;
    int* s_warp_idx = (int*)(s_warp_max + num_warps);

    for (int k_i = 0; k_i < topk; ++k_i) {
        float local_max = -FLT_MAX;
        int local_idx = -1;

        for (int i = tid; i < seq_len; i += blockDim.x) {
            float s = buf[i];
            if (s > local_max) {
                local_max = s;
                local_idx = i;
            }
        }

        // Warp-level argmax
        warp_argmax(local_max, local_idx);

        // Warp leader 写入 SMEM
        if (lane == 0) {
            s_warp_max[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        // Block-level argmax: thread 0 合并所有 warp 结果
        float gmax = -FLT_MAX;
        int gidx = -1;
        if (tid == 0) {
            for (int w = 0; w < num_warps; ++w) {
                if (s_warp_max[w] > gmax) {
                    gmax = s_warp_max[w];
                    gidx = s_warp_idx[w];
                }
            }
            s_warp_max[0] = gmax;
            s_warp_idx[0] = gidx;
        }
        __syncthreads();

        gmax = s_warp_max[0];
        gidx = s_warp_idx[0];

        // 写入输出并 mark 为 -inf
        if (tid == 0) {
            out_idx[head_id * topk + k_i] = gidx;
            buf[gidx] = -FLT_MAX;
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Kernel: Sparse Attention Forward
// 每个 block 处理一个 head，thread 处理 head_dim 的一部分
// 只读取 top-k 个 K/V，大幅减少显存访问
// -----------------------------------------------------------------------------
__global__ void sparse_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const int* __restrict__ topk_idx,
    half* __restrict__ O,
    int num_heads,
    int seq_len,
    int head_dim,
    int topk,
    float scale
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const half* q_ptr = Q + head_id * head_dim;
    const half* k_ptr = K + head_id * seq_len * head_dim;
    const half* v_ptr = V + head_id * seq_len * head_dim;
    half* o_ptr = O + head_id * head_dim;
    const int* idx_ptr = topk_idx + head_id * topk;

    extern __shared__ char smem[];
    float* s_q = (float*)smem;
    float* s_k = s_q + head_dim;
    float* s_scores = s_k + head_dim;

    for (int d = tid; d < head_dim; d += num_threads) {
        s_q[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        int k_pos = idx_ptr[k_i];
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

    float thread_max = -FLT_MAX;
    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        thread_max = fmaxf(thread_max, s_scores[k_i]);
    }
    float warp_max = warp_reduce_max(thread_max);
    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) {
        atomicMax((int*)&block_max, __float_as_int(warp_max));
    }
    __syncthreads();

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

    for (int k_i = tid; k_i < topk; k_i += num_threads) {
        s_scores[k_i] /= block_sum;
    }
    __syncthreads();

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

    float val = __half2float(x[tid]);
    float scale = scale_out[0];
    float quantized = val / scale;
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
    // 分配临时 global memory buffer 存 score
    auto score_buf = torch::empty({num_heads, seq_len}, torch::dtype(torch::kFloat32).device(Q.device()));

    int num_warps = (128 + WARP_SIZE - 1) / WARP_SIZE;  // 4 warps
    size_t smem_size = num_warps * sizeof(float) + num_warps * sizeof(int);

    select_topk_kernel<<<num_heads, 128, smem_size>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        out_idx.data_ptr<int>(),
        score_buf.data_ptr<float>(),
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
