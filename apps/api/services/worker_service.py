from __future__ import annotations

import httpx


async def forward_webrtc_offer(
    worker_base: str,
    session_id: str,
    sdp: str,
    type_: str,
) -> dict[str, str]:
    url = f"{worker_base.rstrip('/')}/webrtc/{session_id}/offer"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json={"sdp": sdp, "type": type_})
        r.raise_for_status()
        return r.json()
