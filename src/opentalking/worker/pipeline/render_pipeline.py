from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from opentalking.core.interfaces.model_adapter import ModelAdapter
from opentalking.core.types.frames import AudioChunk, VideoFrameData

log = logging.getLogger(__name__)


class VideoSink(Protocol):
    async def __call__(self, frame: VideoFrameData) -> None: ...


async def render_audio_chunk(
    adapter: ModelAdapter,
    avatar_state: Any,
    chunk: AudioChunk,
    *,
    frame_index_start: int,
    video_sink: VideoSink,
) -> int:
    """Process one audio chunk through model; returns next frame index."""
    log.info("render_audio_chunk: start (chunk_dur=%.0fms, idx=%d)", chunk.duration_ms, frame_index_start)
    t0 = time.perf_counter()
    feats = adapter.extract_features(chunk)
    t1 = time.perf_counter()
    preds = adapter.infer(feats, avatar_state)
    t2 = time.perf_counter()
    idx = frame_index_start
    for p in preds:
        vf = adapter.compose_frame(avatar_state, idx, p)
        await video_sink(vf)
        idx += 1
    t3 = time.perf_counter()
    n = idx - frame_index_start
    log.info(
        "render_chunk: feat=%.0fms infer=%.0fms compose=%.0fms frames=%d",
        (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, n,
    )
    return idx
