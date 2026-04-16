from __future__ import annotations

import asyncio
import fractions
import time
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av import AudioFrame, VideoFrame

from opentalking.core.types.frames import VideoFrameData

try:
    from aiortc.mediastreams import MediaStreamTrack
except ImportError:  # pragma: no cover
    from aiortc import MediaStreamTrack  # type: ignore


class _NumpyVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: float = 25.0) -> None:
        super().__init__()
        self._fps = fps
        self._interval = 1.0 / fps
        self._queue: asyncio.Queue[VideoFrameData | None] = asyncio.Queue(maxsize=256)
        self._frame_count = 0
        self._next_send: float = 0.0
        self._pacing = False

    async def put(self, frame: VideoFrameData | None) -> None:
        await self._queue.put(frame)

    def reset_clock(self) -> None:
        """Reset pacing clock — call when speech starts so A/V align."""
        self._next_send = time.monotonic()
        self._pacing = True

    async def recv(self) -> VideoFrame:
        item = await self._queue.get()
        if item is None:
            raise asyncio.CancelledError

        if self._pacing:
            now = time.monotonic()
            if now < self._next_send:
                await asyncio.sleep(self._next_send - now)
            self._next_send += self._interval
            # If fallen behind, catch up
            now2 = time.monotonic()
            if self._next_send < now2 - self._interval * 2:
                self._next_send = now2

        vf = VideoFrame.from_ndarray(item.data, format="bgr24")
        self._frame_count += 1
        vf.pts = self._frame_count
        vf.time_base = fractions.Fraction(1, int(max(1, round(self._fps))))
        return vf


class _PCM16AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=512)
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, sample_rate)
        self._next_send: float = 0.0
        self._pacing = False

    async def put_pcm(self, samples: np.ndarray | None) -> None:
        await self._queue.put(samples)

    def reset_clock(self) -> None:
        """Reset pacing clock — call when speech starts so A/V align."""
        self._next_send = time.monotonic()
        self._pacing = True

    async def recv(self) -> AudioFrame:
        samples = await self._queue.get()
        if samples is None:
            raise asyncio.CancelledError
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        n = int(samples.shape[0])

        if self._pacing:
            chunk_duration = n / self.sample_rate
            now = time.monotonic()
            if now < self._next_send:
                await asyncio.sleep(self._next_send - now)
            self._next_send += chunk_duration
            now2 = time.monotonic()
            if self._next_send < now2 - chunk_duration * 2:
                self._next_send = now2

        frame = AudioFrame(format="s16", layout="mono", samples=n)
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = self._time_base
        self._timestamp += n
        return frame


class WebRTCSession:
    """Wraps RTCPeerConnection with numpy video/audio queues."""

    def __init__(self, *, fps: float = 25.0, sample_rate: int = 16000) -> None:
        self.pc = RTCPeerConnection()
        self.video = _NumpyVideoTrack(fps=fps)
        self.audio = _PCM16AudioTrack(sample_rate=sample_rate)
        self.pc.addTrack(self.video)
        self.pc.addTrack(self.audio)

    def reset_clocks(self) -> None:
        """Synchronize pacing clocks and keep media timelines aligned."""
        now = time.monotonic()
        video_time = self.video._frame_count / self.video._fps if self.video._fps > 0 else 0.0
        audio_time = (
            self.audio._timestamp / self.audio.sample_rate
            if self.audio.sample_rate > 0
            else 0.0
        )
        shared_time = max(video_time, audio_time)
        if self.video._fps > 0:
            self.video._frame_count = max(
                self.video._frame_count,
                int(round(shared_time * self.video._fps)),
            )
        if self.audio.sample_rate > 0:
            self.audio._timestamp = max(
                self.audio._timestamp,
                int(round(shared_time * self.audio.sample_rate)),
            )
        self.video._next_send = now
        self.video._pacing = True
        self.audio._next_send = now
        self.audio._pacing = True

    def clear_media_queues(self) -> None:
        """Drop any queued media so the next speech chunk starts cleanly."""
        while True:
            try:
                self.video._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                self.audio._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def handle_offer(self, sdp: str, type_: str) -> RTCSessionDescription:
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription  # type: ignore[return-value]

    @staticmethod
    def _put_close_sentinel(q: asyncio.Queue) -> None:
        """Close tracks without blocking when no peer is draining queued media."""
        try:
            q.put_nowait(None)
            return
        except asyncio.QueueFull:
            pass

        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def close(self) -> None:
        self._put_close_sentinel(self.video._queue)
        self._put_close_sentinel(self.audio._queue)
        await self.pc.close()


def attach_blackhole(pc: RTCPeerConnection) -> MediaBlackhole:
    """Optional debug helper (consume remote tracks)."""
    return MediaBlackhole()
