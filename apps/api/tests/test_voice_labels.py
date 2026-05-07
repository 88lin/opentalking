from __future__ import annotations

from types import SimpleNamespace

import apps.api.routes.voices as voices_routes


def test_dedupe_display_label_keeps_unique_label(monkeypatch):
    monkeypatch.setattr(voices_routes, "list_voices", lambda provider: [])

    assert (
        voices_routes._dedupe_display_label(
            "我的复刻音色",
            provider="dashscope",
            target_model="qwen3-tts-vc-realtime-2026-01-15",
        )
        == "我的复刻音色"
    )


def test_dedupe_display_label_adds_timestamp_for_duplicate(monkeypatch):
    monkeypatch.setattr(
        voices_routes,
        "list_voices",
        lambda provider: [
            SimpleNamespace(
                source="clone",
                target_model="qwen3-tts-vc-realtime-2026-01-15",
                display_label="我的复刻音色",
            )
        ],
    )

    label = voices_routes._dedupe_display_label(
        "我的复刻音色",
        provider="dashscope",
        target_model="qwen3-tts-vc-realtime-2026-01-15",
    )

    assert label.startswith("我的复刻音色-")
    assert label != "我的复刻音色"
