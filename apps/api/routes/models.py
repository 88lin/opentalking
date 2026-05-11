from __future__ import annotations

from fastapi import APIRouter, Request

from opentalking.providers.synthesis.availability import resolve_model_statuses

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_registered_models(request: Request) -> dict[str, list[dict[str, str | bool]] | list[str]]:
    statuses = await resolve_model_statuses(request.app.state.settings)
    return {
        "models": [status.id for status in statuses],
        "statuses": [status.to_dict() for status in statuses],
    }
