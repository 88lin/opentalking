"""FlashTalk engine integration.

This module wraps the 14B diffusion inference pipeline.  All heavy
dependencies (torch, diffusers, etc.) are loaded lazily so that importing
the *package name* alone never triggers a GPU library import.  Use
``from opentalking.engine import get_pipeline`` (or any other public
function) to pull in the real implementation on demand.
"""

__all__ = [
    "get_audio_embedding",
    "get_base_data",
    "get_pipeline",
    "infer_params",
    "run_pipeline",
    "run_pipeline_deferred",
    "run_pipeline_stream",
]


def __getattr__(name: str):
    if name in __all__:
        from opentalking.engine import inference as _inf
        return getattr(_inf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
