from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import opentalking.models  # noqa: F401
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from opentalking.worker.session_runner import SessionRunner
from opentalking.worker.task_consumer import consume_task_queue

runners: dict[str, SessionRunner] = {}


class OfferBody(BaseModel):
    sdp: str
    type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.environ.get("OPENTALKING_REDIS_URL", "redis://localhost:6379/0")
    r = redis.from_url(url, decode_responses=True)
    app.state.redis = r
    avatars = Path(os.environ.get("OPENTALKING_AVATARS_DIR", "./examples/avatars")).resolve()
    app.state.avatars_root = avatars
    device = os.environ.get("OPENTALKING_TORCH_DEVICE", "cuda")
    app.state.device = device
    consumer = asyncio.create_task(consume_task_queue(r, avatars, device, runners))
    yield
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    for s in list(runners.values()):
        await s.close()
    runners.clear()
    await r.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="OpenTalking Worker", lifespan=lifespan)

    @app.post("/webrtc/{session_id}/offer")
    async def webrtc_offer(session_id: str, body: OfferBody, request: Request) -> dict[str, str]:
        runner = runners.get(session_id)
        if not runner:
            raise HTTPException(status_code=404, detail="session not loaded on this worker")
        return await runner.handle_webrtc_offer(body.sdp, body.type)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
