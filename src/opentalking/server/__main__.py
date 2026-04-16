import argparse
import os
from pathlib import Path

import torch.distributed as dist
from loguru import logger

from opentalking.engine import get_pipeline
from opentalking.server import runtime
from opentalking.server.broadcast import CMD_SHUTDOWN, _init_bcast_device, broadcast_cmd
from opentalking.server.idle_cache import _preload_idle_cache_for_ref, _run_startup_warmup
from opentalking.server.worker_loop import worker_loop
from opentalking.server.ws_server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashTalk WebSocket Server")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="models/SoulX-FlashTalk-14B",
        help="FlashTalk model checkpoint directory",
    )
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default="models/chinese-wav2vec2-base",
        help="wav2vec checkpoint directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket server bind address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload for low VRAM usage",
    )
    parser.add_argument(
        "--t5_quant",
        type=str,
        default=os.environ.get("FLASHTALK_T5_QUANT", "").strip() or None,
        choices=["int8", "fp8"],
        help="Optional T5 quantization mode. Defaults to FLASHTALK_T5_QUANT when set.",
    )
    parser.add_argument(
        "--t5_quant_dir",
        type=str,
        default=os.environ.get("FLASHTALK_T5_QUANT_DIR", "").strip() or None,
        help="Directory containing t5_<quant>.safetensors and t5_map_<quant>.json. Defaults to ckpt_dir.",
    )
    parser.add_argument(
        "--wan_quant",
        type=str,
        default=os.environ.get("FLASHTALK_WAN_QUANT", "").strip() or None,
        choices=["int8", "fp8"],
        help="Experimental WanModel weight-only quantization mode.",
    )
    parser.add_argument(
        "--wan_quant_include",
        type=str,
        default=os.environ.get("FLASHTALK_WAN_QUANT_INCLUDE", "").strip() or None,
        help="Comma-separated allowlist for WanModel submodule names.",
    )
    parser.add_argument(
        "--wan_quant_exclude",
        type=str,
        default=os.environ.get("FLASHTALK_WAN_QUANT_EXCLUDE", "").strip() or None,
        help="Comma-separated denylist for WanModel submodule names.",
    )
    args = parser.parse_args()
    if args.t5_quant is not None and args.t5_quant_dir is None:
        args.t5_quant_dir = args.ckpt_dir

    logger.info(
        "[Rank {}/{}] Loading FlashTalk pipeline (ckpt={}, wav2vec={}, t5_quant={}, wan_quant={})",
        runtime.RANK,
        runtime.WORLD_SIZE,
        args.ckpt_dir,
        args.wav2vec_dir,
        args.t5_quant,
        args.wan_quant,
    )
    logger.info(
        "[Rank {}] Params: frame_num={}, motion_frames_num={}, slice_len={}, resolution={}x{}, fps={}",
        runtime.RANK,
        runtime.FRAME_NUM,
        runtime.MOTION_FRAMES_NUM,
        runtime.SLICE_LEN,
        runtime.HEIGHT,
        runtime.WIDTH,
        runtime.TGT_FPS,
    )

    pipeline = get_pipeline(
        world_size=runtime.WORLD_SIZE,
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        cpu_offload=args.cpu_offload,
        t5_quant=args.t5_quant,
        t5_quant_dir=args.t5_quant_dir,
        wan_quant=args.wan_quant,
        wan_quant_include=args.wan_quant_include,
        wan_quant_exclude=args.wan_quant_exclude,
    )
    logger.info("[Rank {}] Pipeline loaded successfully.", runtime.RANK)
    _init_bcast_device()

    startup_refs: list[str] = []
    if runtime.WARMUP_REF_IMAGE:
        startup_refs.append(runtime.WARMUP_REF_IMAGE)
    for ref in runtime.IDLE_PRELOAD_REFS:
        if ref not in startup_refs:
            startup_refs.append(ref)

    if startup_refs:
        warmup_target = runtime.WARMUP_REF_IMAGE or startup_refs[0]
        for ref in startup_refs:
            ref_path = Path(ref).expanduser()
            if not ref_path.is_absolute():
                ref_path = (Path.cwd() / ref_path).resolve()
            if not ref_path.exists():
                logger.warning("[Startup] Skip missing preload ref: {}", ref_path)
                continue
            _preload_idle_cache_for_ref(
                pipeline,
                str(ref_path),
                runtime.WARMUP_PROMPT,
                runtime.WARMUP_SEED,
            )
            if runtime.WARMUP_ON_STARTUP and ref == warmup_target:
                _run_startup_warmup(
                    pipeline,
                    str(ref_path),
                    runtime.WARMUP_PROMPT,
                    runtime.WARMUP_SEED,
                )
        if runtime.WORLD_SIZE > 1:
            dist.barrier()
    elif runtime.WARMUP_ON_STARTUP:
        logger.warning(
            "[Startup] FLASHTALK_WARMUP=1 but no FLASHTALK_WARMUP_REF_IMAGE/FLASHTALK_IDLE_PRELOAD_REFS configured; skipping warmup."
        )

    if runtime.RANK == 0:
        try:
            run_server(pipeline, args.host, args.port)
        except KeyboardInterrupt:
            logger.info("[Server] Interrupted, shutting down...")
        finally:
            if runtime.WORLD_SIZE > 1:
                broadcast_cmd(CMD_SHUTDOWN)
                dist.barrier()
                dist.destroy_process_group()
    else:
        worker_loop(pipeline)
        if runtime.WORLD_SIZE > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
