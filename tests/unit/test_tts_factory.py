from __future__ import annotations

from types import SimpleNamespace

from opentalking.tts.factory import build_tts_adapter, create_tts_adapter


def _settings(**overrides):
    defaults = {
        "normalized_tts_provider": "edge",
        "tts_voice": "zh-CN-XiaoxiaoNeural",
        "ffmpeg_bin": "ffmpeg",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_tts_adapter_uses_edge_provider():
    adapter = build_tts_adapter(
        sample_rate=16000,
        chunk_ms=20.0,
        settings=_settings(normalized_tts_provider="edge"),
    )
    assert adapter.__class__.__name__ == "EdgeTTSAdapter"


def test_build_tts_adapter_auto_falls_back_without_reference():
    adapter = build_tts_adapter(
        sample_rate=16000,
        chunk_ms=20.0,
        settings=_settings(normalized_tts_provider="auto"),
    )
    assert adapter.__class__.__name__ == "EdgeTTSAdapter"


def test_build_tts_adapter_uses_request_provider_override():
    adapter = build_tts_adapter(
        sample_rate=16000,
        chunk_ms=20.0,
        settings=_settings(normalized_tts_provider="edge", tts_voice="zh-CN-XiaoxiaoNeural"),
        default_voice="Cherry",
        tts_provider="dashscope",
        tts_model="qwen3-tts-flash-realtime",
    )

    assert adapter.__class__.__name__ == "DashScopeQwenTTSAdapter"
    assert adapter.default_voice == "Cherry"
    assert adapter._model == "qwen3-tts-flash-realtime"


def test_create_tts_adapter_builds_elevenlabs_for_request_override(monkeypatch):
    monkeypatch.setenv("OPENTALKING_TTS_ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("OPENTALKING_TTS_ELEVENLABS_VOICE_ID", "env-voice")

    adapter = create_tts_adapter(
        sample_rate=16000,
        chunk_ms=20.0,
        default_voice="request-voice",
        tts_provider="elevenlabs",
    )

    assert adapter.__class__.__name__ == "ElevenLabsTTSAdapter"
    assert adapter.default_voice == "request-voice"
