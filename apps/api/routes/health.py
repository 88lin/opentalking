from __future__ import annotations

from fastapi import APIRouter, Request

from opentalking.core.queue_status import get_flashtalk_queue_status

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/queue/status")
async def queue_status(request: Request) -> dict[str, bool | int]:
    try:
        return await get_flashtalk_queue_status(request.app.state.redis)
    except Exception:
        return {"slot_occupied": False, "queue_size": 0}
