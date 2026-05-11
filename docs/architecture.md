# Architecture (current)

OpenTalking is composed of:

- **apps/api** — FastAPI HTTP entry (sessions, avatars, SSE)
- **apps/unified** — single-process dev runner (no external Redis, in-memory bus)
- **apps/web** — React control console
- **opentalking/** — library code (flat layout at repo root, like FastAPI / pydantic / requests)
  - `core/` — interfaces, registry, types, config, bus
  - `models/` — synthesis client shims (FlashTalk WS / FlashHead WS / HTTP); local inference removed (delegated to omnirt)
  - `tts/`, `stt/`, `llm/`, `rtc/`, `voices/`, `avatars/` — capability adapters and asset loaders
  - `worker/` — task consumer + session/speak pipelines
- **omnirt** — external multimodal inference runtime ([repo](https://github.com/datascale-ai/omnirt))

For the design rationale, target layout, decisions, and migration plan, see
[architecture-review.md](architecture-review.md). The active migration is tracked in
[2026-05-07-architecture-refactor-plan.md](2026-05-07-architecture-refactor-plan.md).

## Process model

- **API process** receives REST/SSE traffic, persists session state to Redis, and pushes work onto a queue.
- **Worker process** consumes the queue, drives the speak pipeline (LLM → TTS → synthesis via omnirt → RTC), and emits subtitle/event updates back through Redis pub/sub.
- **Unified mode** collapses both into one process with an in-memory bus — used for dev and quickstart only.

## Inference

All synthesis (FlashTalk / MuseTalk / Wav2Lip) is delegated to omnirt. The repo
contains thin clients (HTTP/WS) only; model weights, GPU/NPU scheduling, and
batching are omnirt's responsibility.

See [render-pipeline.md](render-pipeline.md) for the speak pipeline detail.
