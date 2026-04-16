import torch
import torch.nn.functional as F

try:
    import torch_npu
    _HAS_NPU_FUSION = hasattr(torch_npu, 'npu_fusion_attention')
except ImportError:
    _HAS_NPU_FUSION = False


class BlockDiagonalMask:
    def __init__(self, q_seqlens, kv_seqlens):
        self.q_seqlens = [int(v) for v in q_seqlens]
        self.kv_seqlens = [int(v) for v in kv_seqlens]

    @classmethod
    def from_seqlens(cls, q_seqlens, kv_seqlens):
        return cls(q_seqlens, kv_seqlens)


class _AttnBias:
    BlockDiagonalMask = BlockDiagonalMask


class _Fmha:
    attn_bias = _AttnBias


fmha = _Fmha()


def _attention(q, k, v):
    # q/k/v: [B, S, N, D] (BSND) — xformers convention
    if _HAS_NPU_FUSION and q.device.type == 'npu':
        # npu_fusion_attention with BSND layout
        out = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=q.shape[2],
            input_layout="BSND",
            scale=q.shape[-1] ** -0.5,
            pre_tockens=65535,
            next_tockens=65535,
        )[0]
        return out
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).contiguous()


def memory_efficient_attention(q, k, v, attn_bias=None, op=None):
    del op
    if attn_bias is None:
        return _attention(q, k, v)

    out = torch.empty_like(q)
    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(attn_bias.q_seqlens, attn_bias.kv_seqlens):
        q_end = q_start + q_len
        kv_end = kv_start + kv_len
        out[:, q_start:q_end] = _attention(
            q[:, q_start:q_end],
            k[:, kv_start:kv_end],
            v[:, kv_start:kv_end],
        )
        q_start = q_end
        kv_start = kv_end
    return out
