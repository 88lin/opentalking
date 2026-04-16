from __future__ import annotations

import pytest

import apps.api.main as api_main
import apps.unified.main as unified_main
from opentalking.core.config import Settings


@pytest.mark.parametrize(
    ("cors_origins", "expected"),
    [
        ("*", ["*"]),
        ("http://a.test,http://b.test", ["http://a.test", "http://b.test"]),
        (["http://a.test", "http://b.test"], ["http://a.test", "http://b.test"]),
    ],
)
def test_create_app_accepts_supported_cors_formats(
    monkeypatch: pytest.MonkeyPatch,
    cors_origins: str | list[str],
    expected: list[str],
) -> None:
    settings = Settings(cors_origins=cors_origins)
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_main, "get_settings", lambda: settings)

    api_app = api_main.create_app()
    unified_app = unified_main.create_app()

    assert api_app.user_middleware[0].kwargs["allow_origins"] == expected
    assert unified_app.user_middleware[0].kwargs["allow_origins"] == expected
