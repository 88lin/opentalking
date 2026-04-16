from __future__ import annotations

import concurrent.futures
import struct

import numpy as np
import torch

from opentalking.engine import run_pipeline_deferred, run_pipeline_stream
from opentalking.engine.accelerator import synchronize
from opentalking.server import runtime
from opentalking.server.broadcast import CMD_GENERATE, broadcast_audio_embedding, broadcast_cmd

MAGIC_AUDIO = runtime.MAGIC_AUDIO
MAGIC_VIDEO = runtime.MAGIC_VIDEO


def _encode_jpeg_frame(args: tuple[np.ndarray, list[int]]) -> bytes:
    frame_rgb, encode_params = args
    import cv2

    bgr_frame = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    ok, buf = cv2.imencode(".jpg", bgr_frame, encode_params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def encode_video_jpegs(video_np: np.ndarray) -> list[bytes]:
    import cv2

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, runtime.JPEG_QUALITY]
    tasks = [(video_np[fi], encode_params) for fi in range(video_np.shape[0])]

    if runtime.JPEG_WORKERS <= 1 or len(tasks) <= 1:
        return [_encode_jpeg_frame(task) for task in tasks]

    if runtime._JPEG_EXECUTOR is None:
        runtime._JPEG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=runtime.JPEG_WORKERS,
            thread_name_prefix="flashtalk-jpeg",
        )
    return list(runtime._JPEG_EXECUTOR.map(_encode_jpeg_frame, tasks))


def decode_jpeg_frame(jpeg_bytes: bytes) -> np.ndarray:
    import cv2

    jpeg_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG decoding failed")
    return bgr


def _submit_jpeg_futures(video_np: np.ndarray) -> list:
    import cv2

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, runtime.JPEG_QUALITY]
    tasks = [(video_np[fi], encode_params) for fi in range(video_np.shape[0])]

    if runtime.JPEG_WORKERS <= 1 or len(tasks) <= 1:
        return [_encode_jpeg_frame(task) for task in tasks]

    if runtime._JPEG_EXECUTOR is None:
        runtime._JPEG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=runtime.JPEG_WORKERS,
            thread_name_prefix="flashtalk-jpeg",
        )
    return [runtime._JPEG_EXECUTOR.submit(_encode_jpeg_frame, task) for task in tasks]


def _collect_jpeg_futures(futures: list) -> list[bytes]:
    if not futures:
        return []
    if isinstance(futures[0], bytes):
        return futures
    return [future.result() for future in futures]


def _run_pipeline_stream_for_audio_embedding(pipeline, audio_embedding: torch.Tensor):
    broadcast_cmd(CMD_GENERATE)
    broadcast_audio_embedding(audio_embedding)
    synchronize()

    frame_idx = 0
    for frame_batch in run_pipeline_stream(pipeline, audio_embedding):
        batch_size = frame_batch.shape[0]
        if frame_idx + batch_size <= runtime.MOTION_FRAMES_NUM:
            frame_idx += batch_size
            continue
        skip = max(0, runtime.MOTION_FRAMES_NUM - frame_idx)
        frame_batch = frame_batch[skip:]
        frame_idx += batch_size
        yield frame_batch.cpu().to(torch.uint8).numpy()

    synchronize()


def _run_pipeline_deferred_for_audio_embedding(
    pipeline,
    audio_embedding: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    broadcast_cmd(CMD_GENERATE)
    broadcast_audio_embedding(audio_embedding)
    synchronize()
    video, cond_frame = run_pipeline_deferred(pipeline, audio_embedding)
    synchronize()
    return video[runtime.MOTION_FRAMES_NUM :], cond_frame


def _run_stream_and_encode(pipeline, audio_embedding: torch.Tensor) -> list[bytes]:
    import cv2

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, runtime.JPEG_QUALITY]
    futures: list[concurrent.futures.Future | bytes] = []

    if runtime.JPEG_WORKERS > 1 and runtime._JPEG_EXECUTOR is None:
        runtime._JPEG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=runtime.JPEG_WORKERS,
            thread_name_prefix="flashtalk-jpeg",
        )

    for frame_batch_np in _run_pipeline_stream_for_audio_embedding(pipeline, audio_embedding):
        for frame_idx in range(frame_batch_np.shape[0]):
            frame_rgb = frame_batch_np[frame_idx]
            if runtime.JPEG_WORKERS > 1:
                assert runtime._JPEG_EXECUTOR is not None
                futures.append(
                    runtime._JPEG_EXECUTOR.submit(_encode_jpeg_frame, (frame_rgb, encode_params))
                )
            else:
                futures.append(_encode_jpeg_frame((frame_rgb, encode_params)))

    if runtime.JPEG_WORKERS > 1:
        return [future.result() for future in futures]  # type: ignore[union-attr]
    return futures  # type: ignore[return-value]


def _send_video_message(websocket, jpeg_parts: list[bytes]) -> tuple[int, int]:
    n_frames = len(jpeg_parts)

    if runtime.PROGRESSIVE_SEND:
        total_bytes = 0
        for jpeg_bytes in jpeg_parts:
            payload = MAGIC_VIDEO + struct.pack("<I", 1) + struct.pack("<I", len(jpeg_bytes)) + jpeg_bytes
            websocket.send(payload)
            total_bytes += len(payload)
        return n_frames, total_bytes

    header = MAGIC_VIDEO + struct.pack("<I", n_frames)
    parts = [header]
    total_bytes = 8
    for jpeg_bytes in jpeg_parts:
        parts.append(struct.pack("<I", len(jpeg_bytes)))
        parts.append(jpeg_bytes)
        total_bytes += 4 + len(jpeg_bytes)
    websocket.send(b"".join(parts))
    return n_frames, total_bytes


__all__ = [
    "MAGIC_AUDIO",
    "MAGIC_VIDEO",
    "_collect_jpeg_futures",
    "_encode_jpeg_frame",
    "_run_pipeline_deferred_for_audio_embedding",
    "_run_pipeline_stream_for_audio_embedding",
    "_run_stream_and_encode",
    "_send_video_message",
    "_submit_jpeg_futures",
    "decode_jpeg_frame",
    "encode_video_jpegs",
]
