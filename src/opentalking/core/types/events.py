from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SpeechStartedEvent:
    session_id: str
    text: str


@dataclass
class SubtitleChunkEvent:
    session_id: str
    text: str
    is_final: bool = False


@dataclass
class SpeechEndedEvent:
    session_id: str


@dataclass
class SessionStateChangedEvent:
    session_id: str
    old_state: str
    new_state: str


@dataclass
class ErrorEvent:
    session_id: str
    code: str
    message: str


def event_to_dict(obj: Any) -> dict[str, Any]:
    """Serialize event dataclass to JSON-friendly dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Not a dataclass: {type(obj)}")
