from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from apps.api.routes import avatars


def _png_bytes() -> bytes:
    out = BytesIO()
    Image.new("RGB", (8, 8), (10, 180, 210)).save(out, format="PNG")
    return out.getvalue()


def test_create_custom_avatar_adds_listed_asset_with_preview(tmp_path):
    base = tmp_path / "base-avatar"
    base.mkdir()
    (base / "preview.png").write_bytes(_png_bytes())
    (base / "reference.png").write_bytes(_png_bytes())
    (base / "manifest.json").write_text(
        json.dumps(
            {
                "id": "base-avatar",
                "name": "Base Avatar",
                "model_type": "flashtalk",
                "fps": 25,
                "sample_rate": 16000,
                "width": 416,
                "height": 704,
                "version": "1.0",
            }
        ),
        encoding="utf-8",
    )

    app = FastAPI()
    app.state.settings = SimpleNamespace(avatars_dir=str(tmp_path))
    app.include_router(avatars.router)
    client = TestClient(app)

    response = client.post(
        "/avatars/custom",
        data={"base_avatar_id": "base-avatar", "name": "我的形象"},
        files={"image": ("avatar.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    created = response.json()
    assert created["id"].startswith("custom-")
    assert created["name"] == "我的形象"
    assert created["model_type"] == "flashtalk"

    custom_dir = tmp_path / created["id"]
    assert (custom_dir / "manifest.json").is_file()
    assert (custom_dir / "preview.png").is_file()
    assert (custom_dir / "reference.png").is_file()

    listed = client.get("/avatars").json()
    assert any(item["id"] == created["id"] and item["name"] == "我的形象" for item in listed)

    preview = client.get(f"/avatars/{created['id']}/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
