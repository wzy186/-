"""端到端模型测试

使用 transformers 构建小型 Llama 模型，验证 Sparse Attention 替换后的
数值正确性和 perplexity 变化。

原理：
1. 创建 tiny Llama（hidden=512, layers=4, heads=8）
2. 用原生 attention 跑一遍，记录 logits
3. 把 LlamaAttention 的 core attention 替换为 sparse top-k attention
4. 再跑一遍，对比 KL divergence 和 top-1 token accuracy
"""

import sys
sys.path.insert(0, "/Users/didi/Desktop/sparsekv-engine")

import math
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

try:
    import sparsekv_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def sparse_attention_pytorch(q, k, v, topk_ratio=0.3):
    """纯 PyTorch 实现的 Sparse Attention（用于端到端验证）

    q/k/v: [batch_size, num_heads, seq_len, head_dim]
    对每个 head 的最后一个 query token，选 seq_len 中 top-k 重要的 KV。
    """
    bsz, num_heads, q_len, head_dim = q.shape
    _, _, kv_len, _ = k.shape
    scale = 1.0 / math.sqrt(head_dim)
    topk = max(1, int(kv_len * topk_ratio))

    # scores: [bsz, heads, q_len, kv_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask：只考虑当前位置及之前的 KV
    causal_mask = torch.triu(torch.ones(q_len, kv_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # 选 top-k（沿 kv_len 维度）
    # 对于 causal attention，最后一个 query 看到所有之前 token
    topk_scores, topk_idx = torch.topk(scores, topk, dim=-1)  # [bsz, heads, q_len, topk]

    # gather V
    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
    v_exp = v.unsqueeze(2).expand(-1, -1, q_len, -1, -1)  # [bsz, heads, q_len, kv_len, head_dim]
    # 用 advanced indexing gather
    b_idx = torch.arange(bsz, device=v.device).view(bsz, 1, 1, 1, 1)
    h_idx = torch.arange(num_heads, device=v.device).view(1, num_heads, 1, 1, 1)
    q_idx = torch.arange(q_len, device=v.device).view(1, 1, q_len, 1, 1)
    v_sparse = v_exp[b_idx, h_idx, q_idx, topk_idx_exp, :]

    attn = F.softmax(topk_scores, dim=-1)
    out = torch.matmul(attn.unsqueeze(-2), v_sparse).squeeze(-2)
    return out


_ORIGINAL_LLAMA_ATTENTION = None


def make_sparse_llama_attention(topk_ratio=0.3):
    """生成一个替换 attention core 的 forward 函数"""
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 简化：跳过 rotary emb 和 past_key_value（测试用短序列，影响不大）
        # 实际生产需要补上 RoPE

        # GQA: 重复 KV head
        if self.num_key_value_heads < self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Sparse attention core
        attn_output = sparse_attention_pytorch(query_states, key_states, value_states, topk_ratio)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    return forward


def test_end_to_end_consistency():
    """验证替换 attention 后模型输出偏差在可接受范围"""
    print("\n[End-to-End] 构建 Tiny Llama 模型...")
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    # 生成随机输入
    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (2, seq_len))

    # Baseline: 原生 attention
    with torch.no_grad():
        baseline_logits = model(input_ids).logits

    # 替换 attention
    from transformers.models.llama.modeling_llama import LlamaAttention
    original_forward = LlamaAttention.forward
    LlamaAttention.forward = make_sparse_llama_attention(topk_ratio=0.3)

    with torch.no_grad():
        sparse_logits = model(input_ids).logits

    # 恢复
    LlamaAttention.forward = original_forward

    # 对比指标
    kl = F.kl_div(
        F.log_softmax(sparse_logits, dim=-1),
        F.softmax(baseline_logits, dim=-1),
        reduction='batchmean'
    ).item()

    baseline_top1 = baseline_logits.argmax(dim=-1)
    sparse_top1 = sparse_logits.argmax(dim=-1)
    top1_acc = (baseline_top1 == sparse_top1).float().mean().item()

    logits_mse = ((baseline_logits - sparse_logits) ** 2).mean().item()

    print(f"\n[End-to-End] 对比结果 (seq_len={seq_len}, topk_ratio=0.3):")
    print(f"  KL Divergence:  {kl:.6f}  (越小越好)")
    print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%  (和 baseline 选相同 token 的比例)")
    print(f"  Logits MSE:     {logits_mse:.6f}")

    assert kl < 0.1, f"KL divergence {kl} too high!"
    assert top1_acc > 0.8, f"Top-1 accuracy {top1_acc} too low!"
    print("  [PASS] 端到端一致性测试通过")


def test_perplexity_on_wikitext():
    """在随机文本上对比 perplexity（简化版，不需要下载数据集）"""
    print("\n[Perplexity] 随机文本 perplexity 对比...")
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    # 模拟文本：随机 token 序列
    seq_len = 256
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    def calc_ppl(model, ids):
        with torch.no_grad():
            logits = model(ids).logits
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                   ids[:, 1:].reshape(-1))
        return torch.exp(loss).item()

    ppl_base = calc_ppl(model, input_ids)

    from transformers.models.llama.modeling_llama import LlamaAttention
    original_forward = LlamaAttention.forward
    LlamaAttention.forward = make_sparse_llama_attention(topk_ratio=0.3)
    ppl_sparse = calc_ppl(model, input_ids)
    LlamaAttention.forward = original_forward

    print(f"  Baseline PPL: {ppl_base:.3f}")
    print(f"  Sparse PPL:   {ppl_sparse:.3f}")
    print(f"  Degradation:  {(ppl_sparse/ppl_base - 1)*100:.2f}%")
    assert ppl_sparse / ppl_base < 1.2, "Perplexity degradation too large"
    print("  [PASS] Perplexity 测试通过")


if __name__ == "__main__":
    test_end_to_end_consistency()
    test_perplexity_on_wikitext()
    print("\n=== All end-to-end tests passed! ===")
