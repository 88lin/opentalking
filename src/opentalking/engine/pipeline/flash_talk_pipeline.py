# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import types
from PIL import Image
from loguru import logger
import time

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

from transformers import Wav2Vec2FeatureExtractor

from opentalking.engine.wan.modules import (CLIPModel, T5EncoderModel, WanVAE)
from opentalking.engine.modules.multitalk_model import WanModel
from opentalking.engine.audio.wav2vec2 import Wav2Vec2Model
from opentalking.engine.audio.loudness import match_and_blend_colors_torch, resize_and_centercrop
from opentalking.engine.accelerator import default_device, device_type, empty_cache, synchronize

# compile models to speedup inference
# Enable torch.compile on NPU via env var FLASHTALK_COMPILE=1
COMPILE_MODEL = device_type() == "cuda" or (device_type() == "npu" and os.environ.get("FLASHTALK_COMPILE", "0") == "1")
COMPILE_VAE = device_type() == "cuda"
# use parallel vae to speedup decode/encode
USE_PARALLEL_VAE = True

def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32 and param.__class__.__name__ not in ['WeightQBytesTensor']:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32 and buf.__class__.__name__ not in ['WeightQBytesTensor']:
                module._buffers[name] = buf.to(param_dtype)

def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class FlashTalkPipeline:
    def __init__(
        self,
        config,
        checkpoint_dir,
        wav2vec_dir,
        device=None,
        use_usp=False,
        cpu_offload=False,
        num_timesteps=1000,
        use_timestep_transform=True,
        t5_quant=None,
        t5_quant_dir=None,
        wan_quant=None,
        wan_quant_include=None,
        wan_quant_exclude=None,
    ):
        r"""
        Initializes the image-to-video generation model components.
        Reference from InfiniteTalkPipeline: https://github.com/MeiGen-AI/InfiniteTalk/blob/main/wan/multitalk.py
        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            wav2vec_dir (`str`):
                Path to directory containing wav2vec checkpoints
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
        """
        self.device = device or default_device()
        self.config = config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.use_usp = use_usp and dist.is_initialized()
        self.param_dtype = config.param_dtype
        self.cpu_offload = cpu_offload and not self.use_usp
        self.t5_quant = t5_quant
        self.t5_quant_dir = t5_quant_dir or checkpoint_dir
        self.wan_quant = wan_quant
        self.wan_quant_include = wan_quant_include
        self.wan_quant_exclude = wan_quant_exclude

        if self.t5_quant is not None or self.wan_quant is not None:
            logger.warning(
                "Quantization options are stored for compatibility, but this migrated pipeline does not yet apply them."
            )

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        
        self.vae = WanVAE(
            vae_path=os.path.join(checkpoint_dir, config.vae_checkpoint),
            dtype=self.param_dtype,
            device=self.device,
            parallel=(USE_PARALLEL_VAE and self.use_usp),
        )

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logger.info(f"Creating WanModel from {checkpoint_dir}")

        self.model = WanModel.from_pretrained(
            checkpoint_dir,
            device_map='cpu' if self.cpu_offload else self.device,
            torch_dtype=self.param_dtype,
        )

        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from opentalking.engine.distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.block_profile = os.environ.get("FLASHTALK_BLOCK_PROFILE", "0") == "1"

        if COMPILE_MODEL and not self.cpu_offload:
            self.model = torch.compile(self.model)
        if COMPILE_VAE and not self.cpu_offload:
            self.vae.encode = torch.compile(self.vae.encode)
            self.vae.decode = torch.compile(self.vae.decode)

        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)

    @torch.no_grad()
    def prepare_params(self,
                        input_prompt,
                        cond_image,
                        target_size,
                        frame_num,
                        motion_frames_num,
                        sampling_steps,
                        seed=None,
                        shift=5.0,
                        color_correction_strength=0.0,
                        ):

        if self.cpu_offload:
            self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)[0]
        if self.cpu_offload:
            self.text_encoder.model.cpu()
            empty_cache()

        self.frame_num = frame_num
        self.motion_frames_num = motion_frames_num

        self.target_h, self.target_w = target_size
        self.lat_h, self.lat_w = self.target_h // self.vae_stride[1], self.target_w // self.vae_stride[2]

        if isinstance(cond_image, str):
            cond_image = Image.open(cond_image).convert("RGB")
        cond_image_tensor = resize_and_centercrop(cond_image, (self.target_h, self.target_w)).to(dtype=self.param_dtype, device=self.device)
        cond_image_tensor = (cond_image_tensor / 255 - 0.5) * 2

        self.cond_image_tensor = cond_image_tensor

        self.color_correction_strength = color_correction_strength
        self.original_color_reference = None
        if self.color_correction_strength > 0.0:
            self.original_color_reference = cond_image_tensor.clone()

        if self.cpu_offload:
            self.clip.model.to(self.device)
        clip_context = self.clip.visual(cond_image_tensor[:, :, -1:, :, :]).to(self.param_dtype)
        if self.cpu_offload:
            self.clip.model.cpu()
            empty_cache()

        video_frames = torch.zeros(1, cond_image_tensor.shape[1], frame_num-cond_image_tensor.shape[2], self.target_h, self.target_w).to(dtype=self.param_dtype, device=self.device)

        padding_frames_pixels_values = torch.concat([cond_image_tensor, video_frames], dim=2)

        if self.cpu_offload:
            self.vae.model.to(self.device)
        y = self.vae.encode(padding_frames_pixels_values)
        common_y = y.unsqueeze(0).to(self.param_dtype)

        # get mask
        msk = torch.ones(1, frame_num, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2).to(self.param_dtype)

        y = torch.concat([msk, common_y], dim=1)


        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # prepare timesteps
        if sampling_steps == 2:
            timesteps = [1000, 500]
        elif sampling_steps == 4:
            timesteps = [1000, 750, 500, 250]
        else:
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            
        timesteps.append(0.)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
        self.timesteps = timesteps

        self.arg_c = {
            'context': [context],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': y,
            'ref_target_masks': None,
        }

        self.latent_motion_frames = self.vae.encode(self.cond_image_tensor)

        if self.cpu_offload:
            self.vae.model.cpu()
            empty_cache()

        return

    @torch.no_grad()
    def preprocess_audio(self, speech_array, sr=16000, fps=25):
        video_length = len(speech_array) * fps / sr

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            logger.error("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        return audio_emb

    def _reset_block_profile(self):
        if not self.block_profile:
            return
        model = getattr(self.model, "_orig_mod", self.model)
        for idx, block in enumerate(model.blocks):
            block._profile_index = idx
            block._profile_totals = {}

    def _log_block_profile(self):
        if not self.block_profile or self.rank != 0:
            return
        model = getattr(self.model, "_orig_mod", self.model)
        names = (
            "self_attn",
            "self_qkv",
            "self_rope",
            "self_usp_attention",
            "self_out_proj",
            "text_cross_attn",
            "audio_cross_attn",
            "ffn",
        )
        component_totals = {name: 0.0 for name in names}
        component_counts = {name: 0 for name in names}
        logger.info("[block-profile] block self_attn text_cross_attn audio_cross_attn ffn total calls")
        for idx, block in enumerate(model.blocks):
            totals = dict(getattr(block, "_profile_totals", {}))
            self_attn_totals = getattr(block.self_attn, "_profile_totals", {})
            for name, (total, count) in self_attn_totals.items():
                prev_total, prev_count = totals.get(name, (0.0, 0))
                totals[name] = (prev_total + total, prev_count + count)
            if not totals:
                continue
            values = {}
            counts = {}
            for name in names:
                total, count = totals.get(name, (0.0, 0))
                values[name] = total
                counts[name] = count
                component_totals[name] += total
                component_counts[name] += count
            block_total = (
                values["self_attn"]
                + values["text_cross_attn"]
                + values["audio_cross_attn"]
                + values["ffn"]
            )
            block_calls = max(counts.values()) if counts else 0
            logger.info(
                "[block-profile] "
                f"{idx:02d} "
                f"{values['self_attn']:.4f}s "
                f"{values['text_cross_attn']:.4f}s "
                f"{values['audio_cross_attn']:.4f}s "
                f"{values['ffn']:.4f}s "
                f"{block_total:.4f}s "
                f"{block_calls}"
            )
        total = (
            component_totals["self_attn"]
            + component_totals["text_cross_attn"]
            + component_totals["audio_cross_attn"]
            + component_totals["ffn"]
        )
        logger.info(
            "[block-profile] total "
            f"self_attn={component_totals['self_attn']:.4f}s/{component_counts['self_attn']} "
            f"self_qkv={component_totals['self_qkv']:.4f}s/{component_counts['self_qkv']} "
            f"self_rope={component_totals['self_rope']:.4f}s/{component_counts['self_rope']} "
            f"self_usp_attention={component_totals['self_usp_attention']:.4f}s/{component_counts['self_usp_attention']} "
            f"self_out_proj={component_totals['self_out_proj']:.4f}s/{component_counts['self_out_proj']} "
            f"text_cross_attn={component_totals['text_cross_attn']:.4f}s/{component_counts['text_cross_attn']} "
            f"audio_cross_attn={component_totals['audio_cross_attn']:.4f}s/{component_counts['audio_cross_attn']} "
            f"ffn={component_totals['ffn']:.4f}s/{component_counts['ffn']} "
            f"grand={total:.4f}s"
        )

    @torch.no_grad()
    def generate(self, audio_embedding):
        if self.cpu_offload:
            self.model.to(self.device)
        # evaluation mode
        with torch.no_grad():

            self.arg_c.update({
                "audio": audio_embedding,
            })
            self._reset_block_profile()
            model_for_cache = getattr(self.model, "_orig_mod", self.model)
            if os.environ.get("FLASHTALK_DIT_BLOCK_CACHE", "0") == "1" and hasattr(model_for_cache, "reset_dit_block_cache"):
                model_for_cache.reset_dit_block_cache()

            # sample videos
            latent = torch.randn(
                16, (self.frame_num - 1) // 4 + 1,
                self.lat_h,
                self.lat_w,
                dtype=self.param_dtype,
                device=self.device,
                generator=self.generator)

            latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            # Step-level noise cache: reuse noise prediction on cached steps
            _cache_interval = int(os.environ.get("FLASHTALK_CACHE_INTERVAL", "0"))
            _cached_noise = None

            for i in range(len(self.timesteps)-1):
                timestep = self.timesteps[i]
                latent_model_input = [latent]

                synchronize()
                start_time = time.time()

                if _cache_interval > 0 and i > 0 and i % _cache_interval != 0 and _cached_noise is not None:
                    # Reuse cached noise prediction
                    noise_pred_cond = _cached_noise
                    if self.rank == 0:
                        print(f'[generate] step {i}: reusing cached noise (skip model)')
                else:
                    # inference without CFG
                    if hasattr(model_for_cache, "_dit_block_cache_step"):
                        model_for_cache._dit_block_cache_step = i
                    else:
                        setattr(model_for_cache, "_dit_block_cache_step", i)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **self.arg_c)[0]
                    _cached_noise = noise_pred_cond

                synchronize()
                end_time = time.time()
                if self.rank == 0:
                    print(f'[generate] model denoise per step: {end_time - start_time}s')

                noise_pred = -noise_pred_cond

                # update latent
                t_i = self.timesteps[i][:, None, None, None] / self.num_timesteps
                t_i_1 = self.timesteps[i+1][:, None, None, None] / self.num_timesteps
                x_0 = latent + noise_pred * t_i

                latent = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator)

                latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            if os.environ.get("FLASHTALK_DIT_BLOCK_CACHE", "0") == "1" and self.rank == 0:
                cache_hits = sum(getattr(block, "_dit_cache_hits", 0) for block in model_for_cache.blocks)
                cache_misses = sum(getattr(block, "_dit_cache_misses", 0) for block in model_for_cache.blocks)
                print(f"[generate] dit block cache: hits={cache_hits}, misses={cache_misses}")

            self._log_block_profile()

            if self.cpu_offload:
                self.model.cpu()
                empty_cache()
                self.vae.model.to(self.device)

            synchronize()
            start_decode_time = time.time()
            videos = self.vae.decode(latent.to(self.param_dtype))
            synchronize()
            end_decode_time = time.time()
            if self.rank == 0:
                print(f'[generate] decode video frames: {end_decode_time - start_decode_time}s')
        
        synchronize()
        start_color_correction_time = time.time()
        if self.color_correction_strength > 0.0:
            videos = match_and_blend_colors_torch(videos, self.original_color_reference, self.color_correction_strength)

        cond_frame = videos[:, :, -self.motion_frames_num:].to(self.device)
        synchronize()
        end_color_correction_time = time.time()
        if self.rank == 0:
            print(f'[generate] color correction: {end_color_correction_time - start_color_correction_time}s')

        synchronize()
        start_encode_time = time.time()
        self.latent_motion_frames = self.vae.encode(cond_frame)
        synchronize()
        end_encode_time = time.time()
        if self.rank == 0:
            print(f'[generate] encode motion frames: {end_encode_time - start_encode_time}s')

        if self.cpu_offload:
            self.vae.model.cpu()
            empty_cache()

        gen_video_samples = videos #[:, :, self.motion_frames_num:]

        return gen_video_samples[0].to(torch.float32)

    @torch.no_grad()
    def generate_deferred_motion(self, audio_embedding):
        """Like generate() but defers motion-frame VAE encode.

        Returns ``(video_tensor, cond_frame)`` where *video_tensor* is the
        same output as :meth:`generate` and *cond_frame* is the pixel tensor
        that needs to be encoded via :meth:`finalize_motion_frames`.  This
        lets the caller overlap JPEG encoding (CPU) with motion-frame VAE
        encoding (NPU).
        """
        if self.cpu_offload:
            self.model.to(self.device)

        with torch.no_grad():
            self.arg_c.update({"audio": audio_embedding})
            self._reset_block_profile()
            model_for_cache = getattr(self.model, "_orig_mod", self.model)
            if os.environ.get("FLASHTALK_DIT_BLOCK_CACHE", "0") == "1" and hasattr(model_for_cache, "reset_dit_block_cache"):
                model_for_cache.reset_dit_block_cache()

            latent = torch.randn(
                16, (self.frame_num - 1) // 4 + 1,
                self.lat_h, self.lat_w,
                dtype=self.param_dtype, device=self.device,
                generator=self.generator,
            )
            latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            _cache_interval = int(os.environ.get("FLASHTALK_CACHE_INTERVAL", "0"))
            _cached_noise = None

            for i in range(len(self.timesteps) - 1):
                timestep = self.timesteps[i]
                latent_model_input = [latent]
                synchronize()
                start_time = time.time()

                if _cache_interval > 0 and i > 0 and i % _cache_interval != 0 and _cached_noise is not None:
                    noise_pred_cond = _cached_noise
                else:
                    if hasattr(model_for_cache, "_dit_block_cache_step"):
                        model_for_cache._dit_block_cache_step = i
                    else:
                        setattr(model_for_cache, "_dit_block_cache_step", i)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **self.arg_c)[0]
                    _cached_noise = noise_pred_cond

                synchronize()
                end_time = time.time()
                if self.rank == 0:
                    print(f'[generate_deferred] model denoise per step: {end_time - start_time}s')

                noise_pred = -noise_pred_cond
                t_i = self.timesteps[i][:, None, None, None] / self.num_timesteps
                t_i_1 = self.timesteps[i + 1][:, None, None, None] / self.num_timesteps
                x_0 = latent + noise_pred * t_i
                latent = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(
                    x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator,
                )
                latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            self._log_block_profile()

            if self.cpu_offload:
                self.model.cpu()
                empty_cache()
                self.vae.model.to(self.device)

            synchronize()
            start_decode_time = time.time()
            videos = self.vae.decode(latent.to(self.param_dtype))
            synchronize()
            end_decode_time = time.time()
            if self.rank == 0:
                print(f'[generate_deferred] decode video frames: {end_decode_time - start_decode_time}s')

        # Extract cond_frame but do NOT encode yet
        if self.color_correction_strength > 0.0:
            videos = match_and_blend_colors_torch(
                videos, self.original_color_reference, self.color_correction_strength,
            )
        cond_frame = videos[:, :, -self.motion_frames_num:].to(self.device)

        return videos[0].to(torch.float32), cond_frame

    def finalize_motion_frames(self, cond_frame):
        """Encode motion frames for the next chunk.  Call after generate_deferred_motion."""
        synchronize()
        self.latent_motion_frames = self.vae.encode(cond_frame)
        synchronize()
        if self.cpu_offload:
            self.vae.model.cpu()
            empty_cache()

    @torch.no_grad()
    def generate_stream(self, audio_embedding):
        """Streaming variant of generate(): yields decoded frames one-by-one.

        Each yield produces a single pixel-space frame tensor of shape
        ``[1, C, 1, H, W]`` (float32, range [-1, 1]).  After the final frame
        is yielded the ``latent_motion_frames`` attribute is updated exactly as
        in :meth:`generate`, so callers do **not** need to handle motion-frame
        bookkeeping.

        This allows the caller to start post-processing (e.g. JPEG encoding)
        while VAE decoding is still in progress.
        """
        if self.cpu_offload:
            self.model.to(self.device)

        with torch.no_grad():
            self.arg_c.update({"audio": audio_embedding})
            self._reset_block_profile()
            model_for_cache = getattr(self.model, "_orig_mod", self.model)
            if os.environ.get("FLASHTALK_DIT_BLOCK_CACHE", "0") == "1" and hasattr(model_for_cache, "reset_dit_block_cache"):
                model_for_cache.reset_dit_block_cache()

            latent = torch.randn(
                16, (self.frame_num - 1) // 4 + 1,
                self.lat_h, self.lat_w,
                dtype=self.param_dtype, device=self.device,
                generator=self.generator,
            )
            latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            _cache_interval = int(os.environ.get("FLASHTALK_CACHE_INTERVAL", "0"))
            _cached_noise = None

            for i in range(len(self.timesteps) - 1):
                timestep = self.timesteps[i]
                latent_model_input = [latent]

                synchronize()
                start_time = time.time()

                if _cache_interval > 0 and i > 0 and i % _cache_interval != 0 and _cached_noise is not None:
                    noise_pred_cond = _cached_noise
                    if self.rank == 0:
                        print(f'[generate_stream] step {i}: reusing cached noise (skip model)')
                else:
                    if hasattr(model_for_cache, "_dit_block_cache_step"):
                        model_for_cache._dit_block_cache_step = i
                    else:
                        setattr(model_for_cache, "_dit_block_cache_step", i)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **self.arg_c)[0]
                    _cached_noise = noise_pred_cond

                synchronize()
                end_time = time.time()
                if self.rank == 0:
                    print(f'[generate_stream] model denoise per step: {end_time - start_time}s')

                noise_pred = -noise_pred_cond
                t_i = self.timesteps[i][:, None, None, None] / self.num_timesteps
                t_i_1 = self.timesteps[i + 1][:, None, None, None] / self.num_timesteps
                x_0 = latent + noise_pred * t_i
                latent = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(
                    x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator,
                )
                latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            if os.environ.get("FLASHTALK_DIT_BLOCK_CACHE", "0") == "1" and self.rank == 0:
                cache_hits = sum(getattr(block, "_dit_cache_hits", 0) for block in model_for_cache.blocks)
                cache_misses = sum(getattr(block, "_dit_cache_misses", 0) for block in model_for_cache.blocks)
                print(f"[generate_stream] dit block cache: hits={cache_hits}, misses={cache_misses}")

            self._log_block_profile()

            if self.cpu_offload:
                self.model.cpu()
                empty_cache()
                self.vae.model.to(self.device)

            # --- Streaming VAE decode: yield one time-step at a time ---
            synchronize()
            start_decode_time = time.time()

            # Collect last motion_frames_num pixel frames for VAE encode
            # VAE temporal stride is 4, so total decoded frames = frame_num
            # We need the last motion_frames_num decoded frames.
            motion_pixel_frames: list[torch.Tensor] = []
            frame_idx = 0
            total_decoded_frames = self.frame_num  # 33

            for frame_pixels in self.vae.decode_stream(latent.to(self.param_dtype)):
                # frame_pixels: [1, C, 1, H, W], range [-1, 1]
                frame_pixels = frame_pixels.clamp_(-1, 1)

                # How many pixel frames does this yield produce?
                # The low-level VAE decode_stream yields once per latent time
                # step; each latent step produces ~4 pixel frames (temporal
                # stride), EXCEPT the first step which produces 1.  In the
                # distributed 2D stream path each yield is already gathered
                # across ranks and has shape [1, C, n_frames, H, W].
                n_frames_this = frame_pixels.shape[2]

                # Track last motion_frames_num frames for the next-chunk cond
                remaining_to_start = total_decoded_frames - self.motion_frames_num
                for fi in range(n_frames_this):
                    if frame_idx + fi >= remaining_to_start:
                        motion_pixel_frames.append(
                            frame_pixels[:, :, fi:fi + 1].to(self.device)
                        )
                frame_idx += n_frames_this

                yield frame_pixels  # caller gets to work immediately

            synchronize()
            end_decode_time = time.time()
            if self.rank == 0:
                print(f'[generate_stream] decode video frames (stream): {end_decode_time - start_decode_time}s')

        # --- Motion-frame VAE encode (must run after full decode) ---
        if motion_pixel_frames:
            cond_frame = torch.cat(motion_pixel_frames, dim=2).to(self.device)
            if self.color_correction_strength > 0.0:
                cond_frame = match_and_blend_colors_torch(
                    cond_frame, self.original_color_reference, self.color_correction_strength,
                )
            synchronize()
            start_encode_time = time.time()
            self.latent_motion_frames = self.vae.encode(cond_frame)
            synchronize()
            end_encode_time = time.time()
            if self.rank == 0:
                print(f'[generate_stream] encode motion frames: {end_encode_time - start_encode_time}s')

        if self.cpu_offload:
            self.vae.model.cpu()
            empty_cache()
