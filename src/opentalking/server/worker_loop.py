from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger

from opentalking.engine import get_base_data, run_pipeline
from opentalking.engine.accelerator import synchronize
from opentalking.server import runtime
from opentalking.server.broadcast import (
    CMD_GENERATE,
    CMD_INIT,
    CMD_SHUTDOWN,
    broadcast_audio_embedding,
    broadcast_cmd,
    broadcast_string,
)


def worker_loop(pipeline) -> None:
    logger.info("[Rank {}] Worker loop started, waiting for commands...", runtime.RANK)

    while True:
        cmd = broadcast_cmd(-1)

        if cmd == CMD_INIT:
            image_path = broadcast_string()
            prompt = broadcast_string()
            seed_tensor = torch.zeros(1, dtype=torch.long, device=runtime._BCAST_DEVICE)
            dist.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())

            logger.info("[Rank {}] Executing get_base_data (seed={})", runtime.RANK, seed)
            get_base_data(
                pipeline,
                input_prompt=prompt,
                cond_image=image_path,
                base_seed=seed,
            )
            runtime._reset_audio_embedding_shape_cache()
            logger.info("[Rank {}] get_base_data done", runtime.RANK)
        elif cmd == CMD_GENERATE:
            audio_embedding = broadcast_audio_embedding()
            synchronize()
            run_pipeline(pipeline, audio_embedding)
            synchronize()
        elif cmd == CMD_SHUTDOWN:
            logger.info("[Rank {}] Received shutdown command, exiting.", runtime.RANK)
            break
        else:
            logger.warning("[Rank {}] Unknown command: {}", runtime.RANK, cmd)

    if runtime.WORLD_SIZE > 1:
        dist.barrier()


__all__ = ["worker_loop"]
