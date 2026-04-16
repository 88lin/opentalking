from __future__ import annotations

from fastapi import APIRouter

import opentalking.models  # noqa: F401
from apps.api.core.config import get_settings
from opentalking.models.registry import list_available_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_registered_models() -> dict[str, list[str]]:
    settings = get_settings()
    return {"models": list_available_models(flashtalk_mode=settings.normalized_flashtalk_mode)}
