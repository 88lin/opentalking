from __future__ import annotations

import os

import torch
import torch.distributed as dist
from loguru import logger

from opentalking.engine.accelerator import device_type
from opentalking.server import runtime

CMD_GENERATE = runtime.CMD_GENERATE
CMD_INIT = runtime.CMD_INIT
CMD_NONE = runtime.CMD_NONE
CMD_SHUTDOWN = runtime.CMD_SHUTDOWN
RANK = runtime.RANK
WORLD_SIZE = runtime.WORLD_SIZE


def _init_bcast_device() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", str(RANK)))
    runtime._BCAST_DEVICE = torch.device(f"{device_type()}:{local_rank}")
    logger.info("[Rank {}] broadcast device = {}", RANK, runtime._BCAST_DEVICE)


def broadcast_string(s: str | None = None, src: int = 0) -> str:
    if RANK == src:
        encoded = (s or "").encode("utf-8")
        length_tensor = torch.tensor([len(encoded)], dtype=torch.long, device=runtime._BCAST_DEVICE)
    else:
        length_tensor = torch.zeros(1, dtype=torch.long, device=runtime._BCAST_DEVICE)

    dist.broadcast(length_tensor, src=src)
    length = int(length_tensor.item())

    if RANK == src:
        data_tensor = torch.tensor(list(encoded), dtype=torch.uint8, device=runtime._BCAST_DEVICE)
    else:
        data_tensor = torch.zeros(length, dtype=torch.uint8, device=runtime._BCAST_DEVICE)

    dist.broadcast(data_tensor, src=src)
    if RANK != src:
        s = bytes(data_tensor.cpu().tolist()).decode("utf-8")
    return s or ""


def broadcast_cmd(cmd: int) -> int:
    if RANK == 0:
        return runtime._write_command(cmd)
    return runtime._wait_for_command()


def broadcast_audio_embedding(embedding: torch.Tensor | None = None) -> torch.Tensor:
    if runtime._AUDIO_EMBEDDING_SHAPE_CACHED is None:
        if RANK == 0:
            if embedding is None:
                raise ValueError("embedding is required on rank 0")
            shape_list = list(embedding.shape)
            ndim_tensor = torch.tensor(
                [len(shape_list)],
                dtype=torch.long,
                device=runtime._BCAST_DEVICE,
            )
        else:
            ndim_tensor = torch.zeros(1, dtype=torch.long, device=runtime._BCAST_DEVICE)

        dist.broadcast(ndim_tensor, src=0)
        ndim = int(ndim_tensor.item())

        if RANK == 0:
            shape_tensor = torch.tensor(shape_list, dtype=torch.long, device=runtime._BCAST_DEVICE)
        else:
            shape_tensor = torch.zeros(ndim, dtype=torch.long, device=runtime._BCAST_DEVICE)

        dist.broadcast(shape_tensor, src=0)
        runtime._AUDIO_EMBEDDING_SHAPE_CACHED = tuple(int(x) for x in shape_tensor.cpu().tolist())
    elif RANK == 0:
        if embedding is None:
            raise ValueError("embedding is required on rank 0")
        if tuple(int(x) for x in embedding.shape) != runtime._AUDIO_EMBEDDING_SHAPE_CACHED:
            raise ValueError(
                "Audio embedding shape changed within a session: "
                f"{tuple(int(x) for x in embedding.shape)} != {runtime._AUDIO_EMBEDDING_SHAPE_CACHED}"
            )

    shape = runtime._AUDIO_EMBEDDING_SHAPE_CACHED
    if shape is None:
        raise RuntimeError("broadcast_audio_embedding shape cache was not initialized")

    if RANK == 0:
        assert embedding is not None
        data_tensor = embedding.contiguous().to(device=runtime._BCAST_DEVICE, dtype=torch.bfloat16)
    else:
        data_tensor = torch.zeros(shape, dtype=torch.bfloat16, device=runtime._BCAST_DEVICE)

    dist.broadcast(data_tensor, src=0)
    return data_tensor.to(dtype=torch.float32)


def _reset_audio_embedding_shape_cache() -> None:
    runtime._reset_audio_embedding_shape_cache()


__all__ = [
    "CMD_GENERATE",
    "CMD_INIT",
    "CMD_NONE",
    "CMD_SHUTDOWN",
    "RANK",
    "WORLD_SIZE",
    "_init_bcast_device",
    "_reset_audio_embedding_shape_cache",
    "broadcast_audio_embedding",
    "broadcast_cmd",
    "broadcast_string",
]
