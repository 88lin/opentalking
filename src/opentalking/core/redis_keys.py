"""Shared Redis key names for API and worker."""

TASK_QUEUE = "opentalking:task_queue"


def events_channel(session_id: str) -> str:
    return f"opentalking:events:{session_id}"
