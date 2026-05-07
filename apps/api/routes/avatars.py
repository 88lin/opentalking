from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from opentalking.avatars.loader import load_avatar_bundle
from opentalking.avatars.validator import list_avatar_dirs
from apps.api.schemas.avatar import AvatarSummary

router = APIRouter(prefix="/avatars", tags=["avatars"])


def _avatars_root(request: Request) -> Path:
    return Path(request.app.state.settings.avatars_dir).resolve()


def _summary_from_dir(path: Path) -> AvatarSummary:
    b = load_avatar_bundle(path, strict=False)
    m = b.manifest
    return AvatarSummary(
        id=m.id,
        name=m.name,
        model_type=m.model_type,
        width=m.width,
        height=m.height,
    )


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", value.strip()).strip("-_")
    return slug[:32] or "avatar"


def _unique_avatar_id(root: Path, display_name: str) -> str:
    base = _slugify_name(display_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    candidate = f"custom-{base}-{stamp}"
    candidate = candidate[:96].rstrip("-_")
    if not (root / candidate).exists():
        return candidate
    suffix = 1
    while (root / f"{candidate}-{suffix}").exists():
        suffix += 1
    return f"{candidate}-{suffix}"


async def _read_upload_image(upload: UploadFile) -> Image.Image:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large (max 10MB)")
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid image") from exc
    return image.convert("RGB")


def _write_custom_avatar_manifest(base_manifest_path: Path, target_manifest_path: Path, *, avatar_id: str, name: str) -> None:
    raw = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    raw["id"] = avatar_id
    raw["name"] = name
    metadata = dict(raw.get("metadata") or {})
    metadata["custom_avatar"] = True
    metadata["base_avatar_id"] = json.loads(base_manifest_path.read_text(encoding="utf-8")).get("id")
    raw["metadata"] = metadata
    target_manifest_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@router.get("", response_model=list[AvatarSummary])
async def list_avatars(request: Request) -> list[AvatarSummary]:
    root = _avatars_root(request)
    out: list[AvatarSummary] = []
    for d in list_avatar_dirs(root):
        try:
            out.append(_summary_from_dir(d))
        except Exception:  # noqa: BLE001
            continue
    return out


@router.post("/custom", response_model=AvatarSummary)
async def create_custom_avatar(
    request: Request,
    base_avatar_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...),
) -> AvatarSummary:
    display_name = name.strip()
    if not display_name:
        raise HTTPException(status_code=422, detail="name is required")

    root = _avatars_root(request)
    base_dir = (root / base_avatar_id).resolve()
    try:
        base_dir.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid base_avatar_id") from exc
    if not base_dir.is_dir() or not (base_dir / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="base avatar not found")

    avatar_id = _unique_avatar_id(root, display_name)
    target_dir = root / avatar_id
    image_rgb = await _read_upload_image(image)

    try:
        shutil.copytree(
            base_dir,
            target_dir,
            ignore=shutil.ignore_patterns("reference_custom.*"),
        )
        _write_custom_avatar_manifest(
            base_dir / "manifest.json",
            target_dir / "manifest.json",
            avatar_id=avatar_id,
            name=display_name,
        )
        image_rgb.save(target_dir / "preview.png", format="PNG")
        image_rgb.save(target_dir / "reference.png", format="PNG")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"failed to create custom avatar: {exc}") from exc

    return _summary_from_dir(target_dir)


@router.get("/{avatar_id}")
async def get_avatar(avatar_id: str, request: Request) -> AvatarSummary:
    root = _avatars_root(request)
    path = root / avatar_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="avatar not found")
    return _summary_from_dir(path)


@router.get("/{avatar_id}/preview")
async def get_preview(avatar_id: str, request: Request) -> FileResponse:
    root = _avatars_root(request)
    path = root / avatar_id / "preview.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="preview not found")
    return FileResponse(path, media_type="image/png")
