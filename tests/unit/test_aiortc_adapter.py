from __future__ import annotations

from opentalking.rtc.aiortc_adapter import WebRTCSession


def test_reset_clocks_aligns_audio_timeline_to_video() -> None:
    session = WebRTCSession(fps=25.0, sample_rate=16000)
    try:
        session.video._frame_count = 250
        session.audio._timestamp = 0

        session.reset_clocks()

        assert session.video._frame_count == 250
        assert session.audio._timestamp == 160000
        assert session.video._pacing is True
        assert session.audio._pacing is True
    finally:
        session._put_close_sentinel(session.video._queue)
        session._put_close_sentinel(session.audio._queue)


def test_clear_media_queues_drops_buffered_audio_and_video() -> None:
    session = WebRTCSession(fps=25.0, sample_rate=16000)
    try:
        session.video._queue.put_nowait(object())
        session.audio._queue.put_nowait(object())

        session.clear_media_queues()

        assert session.video._queue.qsize() == 0
        assert session.audio._queue.qsize() == 0
    finally:
        session._put_close_sentinel(session.video._queue)
        session._put_close_sentinel(session.audio._queue)
