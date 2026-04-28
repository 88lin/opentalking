"""Shared Redis key names for API and worker."""

TASK_QUEUE = "opentalking:task_queue"
FLASHTALK_QUEUE_STATUS = "opentalking:flashtalk_queue_status"


def events_channel(session_id: str) -> str:
    return f"opentalking:events:{session_id}"


def uploaded_pcm_key(session_id: str, upload_id: str) -> str:
    return f"opentalking:uploaded_pcm:{session_id}:{upload_id}"
