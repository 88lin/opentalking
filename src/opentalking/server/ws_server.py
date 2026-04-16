from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from opentalking.server import runtime
from opentalking.server.broadcast import CMD_INIT, broadcast_cmd, broadcast_string
from opentalking.server.idle_cache import (
    _advance_idle_cache_cursor,
    _apply_idle_region_constraints,
    _build_idle_eye_mask,
    _build_idle_mouth_mask,
    _build_idle_refresh_audio_array,
    _crossfade_frames,
    _generate_idle_cache_frames,
    _load_idle_cache_frames,
    _load_reference_frame,
    _make_idle_cache_key,
    _prepare_audio_embedding_for_chunk,
    _prepare_pipeline_state,
    _render_video_frames_for_audio_embedding,
    _run_session_warmup,
    _sample_idle_hold_chunks,
    _save_idle_cache_frames,
)
from opentalking.server.video_codec import MAGIC_AUDIO, _send_video_message, encode_video_jpegs


@dataclass
class ServerSession:
    active: bool = False
    audio_buffer: np.ndarray | None = None
    audio_write_pos: int = 0
    chunk_idx: int = 0
    temp_image_path: str | None = None
    idle_cache_frames: list[np.ndarray] = field(default_factory=list)
    idle_cache_index: int = 0
    idle_cache_direction: int = 1
    idle_hold_remaining: int = 0
    using_idle_cache: bool = False
    silence_chunk_run: int = 0
    idle_refresh_counter: int = 0
    idle_refresh_generation: int = 0
    idle_reference_frame: np.ndarray | None = None
    idle_mouth_mask: np.ndarray | None = None
    idle_eye_mask: np.ndarray | None = None
    last_idle_locked_frames: np.ndarray | None = None
    last_output_frames: np.ndarray | None = None
    last_chunk_rms: float = 0.0
    idle_rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(runtime.IDLE_RANDOM_SEED)
    )


def _cleanup_temp_image(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _reset_session(session: ServerSession, *, cleanup_temp_image: bool) -> None:
    temp_path = session.temp_image_path
    session.active = False
    session.audio_buffer = None
    session.audio_write_pos = 0
    session.chunk_idx = 0
    session.temp_image_path = None
    session.idle_cache_frames = []
    session.idle_cache_index = 0
    session.idle_cache_direction = 1
    session.idle_hold_remaining = 0
    session.using_idle_cache = False
    session.silence_chunk_run = 0
    session.idle_refresh_counter = 0
    session.idle_refresh_generation = 0
    session.idle_reference_frame = None
    session.idle_mouth_mask = None
    session.idle_eye_mask = None
    session.last_idle_locked_frames = None
    session.last_output_frames = None
    session.last_chunk_rms = 0.0
    session.idle_rng = np.random.default_rng(runtime.IDLE_RANDOM_SEED)
    runtime._reset_audio_embedding_shape_cache()
    if cleanup_temp_image:
        _cleanup_temp_image(temp_path)


def run_server(pipeline, host: str, port: int) -> None:
    from websockets.sync.server import serve as ws_serve

    session = ServerSession()

    def handler(websocket) -> None:
        remote = websocket.request.headers.get("Host", "unknown")
        logger.info("[Server] New connection from {}", remote)

        try:
            for message in websocket:
                if isinstance(message, str):
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "init":
                        if session.active:
                            logger.warning(
                                "[Server] Replacing existing active session with a new init request."
                            )
                            _reset_session(session, cleanup_temp_image=True)

                        ref_image_b64 = msg.get("ref_image", "")
                        prompt = msg.get("prompt", runtime.WARMUP_PROMPT)
                        seed = int(msg.get("seed", 9999))
                        if not ref_image_b64:
                            websocket.send(
                                json.dumps({"type": "error", "message": "Missing 'ref_image' field."})
                            )
                            continue

                        try:
                            image_data = base64.b64decode(ref_image_b64)
                        except Exception:
                            websocket.send(
                                json.dumps(
                                    {"type": "error", "message": "Invalid base64 in 'ref_image'."}
                                )
                            )
                            continue

                        fd, temp_image_path = tempfile.mkstemp(suffix=".png")
                        with os.fdopen(fd, "wb") as handle:
                            handle.write(image_data)
                        session.temp_image_path = temp_image_path

                        logger.info(
                            "[Server] Init: prompt={!r}, seed={}, image={} bytes -> {}",
                            prompt,
                            seed,
                            len(image_data),
                            temp_image_path,
                        )

                        try:
                            broadcast_cmd(CMD_INIT)
                            broadcast_string(temp_image_path)
                            broadcast_string(prompt)
                            seed_tensor = torch.tensor(
                                [seed],
                                dtype=torch.long,
                                device=runtime._BCAST_DEVICE,
                            )
                            dist.broadcast(seed_tensor, src=0)

                            session.idle_reference_frame = _load_reference_frame(temp_image_path)
                            session.idle_mouth_mask = _build_idle_mouth_mask(
                                runtime.HEIGHT,
                                runtime.WIDTH,
                            )
                            session.idle_eye_mask = _build_idle_eye_mask(
                                runtime.HEIGHT,
                                runtime.WIDTH,
                            )
                            idle_cache_key = _make_idle_cache_key(session.idle_reference_frame)
                            _prepare_pipeline_state(pipeline, temp_image_path, prompt, seed)
                            session.idle_cache_frames = _load_idle_cache_frames(idle_cache_key) or []
                            if not session.idle_cache_frames:
                                session.idle_cache_frames = _generate_idle_cache_frames(pipeline)
                                if session.idle_cache_frames:
                                    _save_idle_cache_frames(idle_cache_key, session.idle_cache_frames)
                                _prepare_pipeline_state(pipeline, temp_image_path, prompt, seed)
                            if runtime.WARMUP_ON_INIT:
                                _run_session_warmup(pipeline)
                        except Exception as exc:
                            logger.error("[Server] Init failed: {}", exc)
                            _reset_session(session, cleanup_temp_image=True)
                            websocket.send(
                                json.dumps({"type": "error", "message": f"Init failed: {exc}"})
                            )
                            continue

                        session.active = True
                        session.audio_buffer = np.zeros(runtime.CACHED_AUDIO_SAMPLES, dtype=np.float32)
                        session.idle_rng = np.random.default_rng(runtime.IDLE_RANDOM_SEED + seed)
                        session.idle_hold_remaining = _sample_idle_hold_chunks(
                            session.idle_rng,
                            len(session.idle_cache_frames),
                        )
                        websocket.send(
                            json.dumps(
                                {
                                    "type": "init_ok",
                                    "frame_num": runtime.FRAME_NUM,
                                    "motion_frames_num": runtime.MOTION_FRAMES_NUM,
                                    "slice_len": runtime.SLICE_LEN,
                                    "fps": runtime.TGT_FPS,
                                    "height": runtime.HEIGHT,
                                    "width": runtime.WIDTH,
                                }
                            )
                        )
                        logger.info("[Server] Init OK, session active.")
                    elif msg_type == "close":
                        logger.info("[Server] Client requested close.")
                        _reset_session(session, cleanup_temp_image=True)
                        websocket.send(json.dumps({"type": "close_ok"}))
                    else:
                        websocket.send(
                            json.dumps(
                                {"type": "error", "message": f"Unknown message type: {msg_type}"}
                            )
                        )
                elif isinstance(message, bytes):
                    if not session.active or session.audio_buffer is None:
                        websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "No active session. Send 'init' first.",
                                }
                            )
                        )
                        continue
                    if len(message) < 4 or message[:4] != MAGIC_AUDIO:
                        websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "Binary message must start with 'AUDI' magic.",
                                }
                            )
                        )
                        continue

                    pcm_bytes = message[4:]
                    if len(pcm_bytes) != runtime.AUDIO_CHUNK_BYTES:
                        websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": (
                                        f"Expected {runtime.AUDIO_CHUNK_BYTES} bytes of int16 PCM "
                                        f"({runtime.AUDIO_CHUNK_SAMPLES} samples), got {len(pcm_bytes)}."
                                    ),
                                }
                            )
                        )
                        continue

                    chunk_audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    session.last_chunk_rms = runtime._chunk_rms(chunk_audio)
                    session.silence_chunk_run = (
                        session.silence_chunk_run + 1
                        if session.last_chunk_rms <= runtime.IDLE_SILENCE_RMS
                        else 0
                    )

                    session.audio_write_pos = runtime._append_audio_chunk(
                        session.audio_buffer,
                        session.audio_write_pos,
                        chunk_audio,
                    )
                    audio_array = runtime._linearize_audio_buffer(
                        session.audio_buffer,
                        session.audio_write_pos,
                    )

                    t_start = time.time()
                    previous_mode = "idle" if session.using_idle_cache else "live"
                    current_mode = previous_mode
                    selected_idle_slot: int | None = None

                    try:
                        if (
                            session.idle_cache_frames
                            and session.silence_chunk_run >= runtime.IDLE_ENTER_CHUNKS
                        ):
                            selected_idle_slot = session.idle_cache_index
                            should_refresh_idle = (
                                session.using_idle_cache
                                and session.idle_refresh_counter >= runtime.IDLE_REFRESH_INTERVAL
                            )
                            if should_refresh_idle:
                                current_mode = "idle_refresh"
                                refresh_audio_array = _build_idle_refresh_audio_array(
                                    session.audio_buffer,
                                    session.audio_write_pos,
                                    runtime.IDLE_RANDOM_SEED
                                    + 1000
                                    + session.idle_refresh_generation * 23,
                                )
                                audio_embedding = _prepare_audio_embedding_for_chunk(
                                    pipeline,
                                    refresh_audio_array,
                                )
                                video_np = _render_video_frames_for_audio_embedding(
                                    pipeline,
                                    audio_embedding,
                                )
                                session.idle_refresh_counter = 0
                                session.idle_refresh_generation += 1
                            else:
                                current_mode = "idle"
                                video_np = session.idle_cache_frames[selected_idle_slot].copy()
                                if session.using_idle_cache:
                                    session.idle_refresh_counter += 1

                            session.idle_hold_remaining -= 1
                            if session.idle_hold_remaining <= 0:
                                (
                                    session.idle_cache_index,
                                    session.idle_cache_direction,
                                ) = _advance_idle_cache_cursor(
                                    session.idle_cache_index,
                                    session.idle_cache_direction,
                                    len(session.idle_cache_frames),
                                )
                                session.idle_hold_remaining = _sample_idle_hold_chunks(
                                    session.idle_rng,
                                    len(session.idle_cache_frames),
                                )
                        else:
                            current_mode = "live"
                            audio_embedding = _prepare_audio_embedding_for_chunk(pipeline, audio_array)
                            video_np = _render_video_frames_for_audio_embedding(
                                pipeline,
                                audio_embedding,
                            )
                            session.idle_hold_remaining = _sample_idle_hold_chunks(
                                session.idle_rng,
                                len(session.idle_cache_frames),
                            )
                            session.idle_refresh_counter = 0
                    except Exception as exc:
                        logger.error("[Server] Generate failed at chunk {}: {}", session.chunk_idx, exc)
                        websocket.send(
                            json.dumps({"type": "error", "message": f"Generate failed: {exc}"})
                        )
                        continue

                    t_infer = time.time()
                    if current_mode == "idle" or previous_mode == "idle":
                        video_np = _crossfade_frames(
                            session.last_output_frames,
                            video_np,
                            runtime.IDLE_CACHE_CROSSFADE_FRAMES,
                        )

                    if current_mode != "live":
                        video_np = _apply_idle_region_constraints(
                            video_np,
                            session.idle_reference_frame,
                            session.idle_mouth_mask,
                            session.last_idle_locked_frames,
                            runtime.IDLE_MOUTH_LOCK,
                            runtime.IDLE_MOUTH_TEMPORAL,
                        )
                        video_np = _apply_idle_region_constraints(
                            video_np,
                            session.idle_reference_frame,
                            session.idle_eye_mask,
                            session.last_idle_locked_frames,
                            runtime.IDLE_EYE_LOCK,
                            runtime.IDLE_EYE_TEMPORAL,
                        )
                        session.last_idle_locked_frames = video_np.copy()
                        if selected_idle_slot is not None:
                            session.idle_cache_frames[selected_idle_slot] = video_np.copy()
                    else:
                        session.last_idle_locked_frames = None

                    jpeg_parts = encode_video_jpegs(video_np)
                    t_encode = time.time()
                    n_frames, total_bytes = _send_video_message(websocket, jpeg_parts)
                    t_send = time.time()
                    session.last_output_frames = video_np
                    session.using_idle_cache = current_mode != "live"
                    logger.info(
                        "[Server] chunk-{}: {}f, infer={:.2f}s encode={:.2f}s send={:.2f}s "
                        "total={:.2f}s mode={} rms={:.5f} silent_run={} size={}KB "
                        "jpeg_q={} jpeg_workers={}",
                        session.chunk_idx,
                        n_frames,
                        t_infer - t_start,
                        t_encode - t_infer,
                        t_send - t_encode,
                        t_send - t_start,
                        current_mode,
                        session.last_chunk_rms,
                        session.silence_chunk_run,
                        total_bytes // 1024,
                        runtime.JPEG_QUALITY,
                        runtime.JPEG_WORKERS,
                    )
                    session.chunk_idx += 1
        except Exception as exc:
            logger.warning("[Server] Connection error: {}", exc)
        finally:
            if session.active:
                logger.info("[Server] Client disconnected, cleaning up session.")
                _reset_session(session, cleanup_temp_image=True)

    logger.info("[Server] Rank 0 WebSocket server starting on {}:{}", host, port)
    with ws_serve(handler, host, port, max_size=50 * 1024 * 1024) as server:
        server.serve_forever()


__all__ = ["ServerSession", "run_server"]
