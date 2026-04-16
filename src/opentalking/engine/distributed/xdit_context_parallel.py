# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
import os
import time
try:
    import torch_npu  # noqa: F401
    _amp_device = "npu"
except ImportError:
    _amp_device = "cuda"
import torch.amp as amp
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from einops import rearrange
_IS_NPU = (_amp_device == "npu")
if _IS_NPU:
    from yunchang import UlyssesAttention
    from yunchang.kernels import AttnType as _YCAttnType
    _usp_attn_instance = None
else:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention


def _usp_attention(q, k, v, window_size=(-1, -1)):
    """Device-agnostic USP attention wrapper."""
    if _IS_NPU:
        global _usp_attn_instance
        if _usp_attn_instance is None:
            _usp_attn_instance = UlyssesAttention(
                sequence_process_group=get_sp_group().device_group,
                attn_type=_YCAttnType.NPU,
            )
        return _usp_attn_instance(query=q, key=k, value=v, window_size=window_size)
    else:
        return xFuserLongContextAttention()(None, query=q, key=k, value=v, window_size=window_size)
try:
    import xformers.ops as attention_ops
except (ImportError, ModuleNotFoundError):
    from opentalking.engine import attention_ops

from opentalking.engine.modules.multitalk_model import sinusoidal_embedding_1d, _parse_block_cache_blocks
from opentalking.engine.audio.loudness import get_attn_map_with_target, split_token_counts_and_frame_ids, normalize_and_scale
from opentalking.engine.modules.multitalk_attention import SingleStreamMutiAttention

_BLOCK_PROFILE = os.environ.get("FLASHTALK_BLOCK_PROFILE", "0") == "1"


def _profile_sync():
    if _amp_device == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_add(module, name, elapsed):
    totals = getattr(module, "_profile_totals", None)
    if totals is None:
        totals = {}
        module._profile_totals = totals
    total, count = totals.get(name, (0.0, 0))
    totals[name] = (total + elapsed, count + 1)


def _dit_block_cache_config(num_layers):
    if os.environ.get("FLASHTALK_DIT_BLOCK_CACHE", "0") != "1":
        return False, set(), 0

    interval = int(os.environ.get("FLASHTALK_DIT_BLOCK_CACHE_INTERVAL", "2"))
    start = int(os.environ.get("FLASHTALK_DIT_BLOCK_CACHE_START", "1"))
    blocks = _parse_block_cache_blocks(
        os.environ.get("FLASHTALK_DIT_BLOCK_CACHE_BLOCKS", "all"),
        num_layers,
    )
    return interval > 1, blocks, start


def _dit_block_cache_apply(self, x, kwargs):
    cache_enabled, cache_blocks, cache_start = _dit_block_cache_config(len(self.blocks))
    if cache_enabled:
        cache_interval = int(os.environ.get("FLASHTALK_DIT_BLOCK_CACHE_INTERVAL", "2"))
        cache_step = int(getattr(self, "_dit_block_cache_step", 0))
        use_cached_blocks = (
            cache_step >= cache_start
            and cache_step % cache_interval != 0
        )
    else:
        use_cached_blocks = False

    for block_idx, block in enumerate(self.blocks):
        cache_key = (tuple(x.shape), x.dtype, x.device.type, x.device.index)
        cached_delta = getattr(block, "_dit_cache_delta", None)
        cached_key = getattr(block, "_dit_cache_key", None)
        if (
            use_cached_blocks
            and block_idx in cache_blocks
            and cached_delta is not None
            and cached_key == cache_key
        ):
            block._dit_cache_hits = getattr(block, "_dit_cache_hits", 0) + 1
            x = x + cached_delta.to(device=x.device, dtype=x.dtype)
            continue

        if cache_enabled and block_idx in cache_blocks:
            block_input = x
        x = block(x, **kwargs)
        if cache_enabled and block_idx in cache_blocks:
            block._dit_cache_delta = (x - block_input).detach()
            block._dit_cache_key = cache_key
            block._dit_cache_misses = getattr(block, "_dit_cache_misses", 0) + 1

    return x

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def pad_freqs_zero(original_tensor, target_len):
    """Pad with zeros for sin component (no rotation for padded positions)."""
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.zeros(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast(_amp_device, enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      (cos, sin) tuple, each [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    freqs_cos, freqs_sin = freqs

    # split freqs
    freqs_cos = freqs_cos.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_sin = freqs_sin.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # reshape x to pairs for rotary embedding
        out_dtype = x.dtype
        x_pair = x[i, :s].float().reshape(s, n, -1, 2)
        x_real = x_pair[..., 0]  # [s, n, c]
        x_imag = x_pair[..., 1]  # [s, n, c]

        # build freq grids (real-valued)
        cos_i = torch.cat([
            freqs_cos[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_cos[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_cos[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        sin_i = torch.cat([
            freqs_sin[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_sin[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_sin[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply sequence parallel slicing
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        cos_i = pad_freqs(cos_i, s * sp_size)
        sin_i = pad_freqs_zero(sin_i, s * sp_size)
        s_per_rank = s
        cos_i_rank = cos_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
        sin_i_rank = sin_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]

        # apply rotary: (x_real + i*x_imag) * (cos + i*sin)
        out_real = x_real * cos_i_rank - x_imag * sin_i_rank
        out_imag = x_real * sin_i_rank + x_imag * cos_i_rank

        x_i = torch.stack([out_real, out_imag], dim=-1).flatten(2).to(out_dtype)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output)


def usp_dit_forward_vace(self, x, vace_context, seq_len, kwargs):
    # embeddings
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in c
    ])

    # arguments
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)

    # Context Parallel
    c = torch.chunk(
        c, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    hints = []
    for block in self.vace_blocks:
        c, c_skip = block(c, **new_kwargs)
        hints.append(c_skip)
    return hints


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    vace_context=None,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs[0].device != device:
        self.freqs = (self.freqs[0].to(device), self.freqs[1].to(device))

    if self.model_type != 'vace' and y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    with amp.autocast(_amp_device, dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float()).float()
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if self.model_type != 'vace' and clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    x = _dit_block_cache_apply(self, x, kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_start = time.time()

    q, k, v = qkv_fn(x)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_qkv", time.time() - _profile_start)
        _profile_start = time.time()

    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_rope", time.time() - _profile_start)
        _profile_start = time.time()

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = _usp_attention(half(q), half(k), half(v), window_size=self.window_size)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_usp_attention", time.time() - _profile_start)
        _profile_start = time.time()

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_out_proj", time.time() - _profile_start)
    return x

def usp_dit_forward_multitalk(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    audio=None,
    ref_target_masks=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    
    assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs[0].device != device:
        self.freqs = (self.freqs[0].to(device), self.freqs[1].to(device))

    _, T, H, W = x[0].shape
    N_t = T // self.patch_size[0]
    N_h = H // self.patch_size[1]
    N_w = W // self.patch_size[2]

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    x[0] = x[0].to(context[0].dtype)

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    with amp.autocast(_amp_device, dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float()).float()
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  
        context = torch.concat([context_clip, context], dim=1)

    # get audio token
    audio_cond = audio.to(device=x.device, dtype=x.dtype)
    first_frame_audio_emb_s = audio_cond[:, :1, ...]
    latter_frame_audio_emb = audio_cond[:, 1:, ...]
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale) 
    middle_index = self.audio_window // 2
    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
    audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) 
    human_num = len(audio_embedding)
    audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)


    # convert ref_target_masks to token_ref_target_masks
    if ref_target_masks is not None:
        ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
        token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest') 
        token_ref_target_masks = token_ref_target_masks.squeeze(0) 
        token_ref_target_masks = (token_ref_target_masks > 0)
        token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
        token_ref_target_masks = token_ref_target_masks.to(x.dtype)
    else:
        token_ref_target_masks = None

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        audio_embedding=audio_embedding,
        ref_target_masks=token_ref_target_masks,
        human_num=human_num,
        )

    x = _dit_block_cache_apply(self, x, kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
        
    return torch.stack(x).float()


def usp_attn_forward_multitalk(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16,
                     ref_target_masks=None):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_start = time.time()

    q, k, v = qkv_fn(x)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_qkv", time.time() - _profile_start)
        _profile_start = time.time()

    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_rope", time.time() - _profile_start)
        _profile_start = time.time()


    x = _usp_attention(half(q), half(k), half(v), window_size=self.window_size)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_usp_attention", time.time() - _profile_start)
        _profile_start = time.time()

    # output
    x = x.flatten(2)
    x = self.o(x)

    if _BLOCK_PROFILE:
        _profile_sync()
        _profile_add(self, "self_out_proj", time.time() - _profile_start)

    if ref_target_masks is not None:
        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(q.type_as(x), k.type_as(x), grid_sizes[0], 
                                                ref_target_masks=ref_target_masks, enable_sp=True)
    else:
        x_ref_attn_map = None

    return x, x_ref_attn_map


def usp_crossattn_multi_forward_multitalk(self, 
                                        x: torch.Tensor, 
                                        encoder_hidden_states: torch.Tensor,  # 1, 21, 64, C
                                        shape=None, 
                                        x_ref_attn_map=None,
                                        human_num=None) -> torch.Tensor:
        
        N_t, N_h, N_w = shape 
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        audio_tokens_per_frame = 32
        visual_seqlen, frame_ids = split_token_counts_and_frame_ids(N_t, N_h * N_w, sp_size, sp_rank)
        encoder_hidden_states = encoder_hidden_states[:, min(frame_ids):max(frame_ids)+1, ...]
        encoder_hidden_states = rearrange(encoder_hidden_states, "B T N C -> B (T N) C")
        N_a = len(frame_ids)
        kv_seq = [audio_tokens_per_frame * human_num] * N_a

        if human_num == 1:
            return super(SingleStreamMutiAttention, self).forward(x, encoder_hidden_states, shape, enable_sp=True, kv_seq=kv_seq)


        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x) 
        q_shape = (B, N, self.num_heads, self.head_dim) 
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)

        max_values = x_ref_attn_map.max(1).values[:, None, None] 
        min_values = x_ref_attn_map.min(1).values[:, None, None] 
        max_min_values = torch.cat([max_values, min_values], dim=2)
        max_min_values = get_sp_group().all_gather(max_min_values, dim=1)

        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
        back   = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] # N 
        q = self.rope_1d(q, normalized_pos)
 
        encoder_kv = self.kv_linear(encoder_hidden_states) 
        encoder_kv_shape = (B, encoder_hidden_states.size(1), 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0) # B H N C

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        # position embedding for condition audio embeddings
        per_frame = torch.zeros(audio_tokens_per_frame * human_num, dtype=encoder_k.dtype).to(encoder_k.device)
        per_frame[:audio_tokens_per_frame] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[audio_tokens_per_frame:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame]*N_a, dim=0)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)

        # get attn
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        attn_bias = attention_ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(visual_seqlen, kv_seq)
        x = attention_ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=attn_bias, op=None,)
        x = rearrange(x, "B M H K -> B H M K")

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape) 
        x = self.proj(x) 
        x = self.proj_drop(x)

        return x
