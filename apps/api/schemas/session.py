from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    avatar_id: str = Field(..., examples=["demo-avatar"])
    model: str = Field(..., examples=["wav2lip"])


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str = "created"


class SpeakRequest(BaseModel):
    text: str


class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
