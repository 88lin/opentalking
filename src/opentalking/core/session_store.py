from __future__ import annotations

from typing import Any

SESSION_TTL_SECONDS = 600
TERMINAL_STATES = {"closed", "error"}


def session_key(session_id: str) -> str:
    return f"opentalking:session:{session_id}"


async def get_session_record(r: Any, session_id: str) -> dict[str, str] | None:
    raw = await r.hgetall(session_key(session_id))
    if not raw:
        return None
    return dict(raw)


async def set_session_state(
    r: Any,
    session_id: str,
    state: str,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    mapping: dict[str, Any] = {"state": state}
    if extra:
        mapping.update(extra)
    await r.hset(session_key(session_id), mapping=mapping)
    if state in TERMINAL_STATES:
        await r.expire(session_key(session_id), SESSION_TTL_SECONDS)
    else:
        persist = getattr(r, "persist", None)
        if callable(persist):
            await persist(session_key(session_id))
