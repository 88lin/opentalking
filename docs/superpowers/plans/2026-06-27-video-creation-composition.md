# Video Creation Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-generation scene composition preview to Video Creation and use the one-off composition settings when exporting generated videos.

**Architecture:** The frontend derives a selected avatar scene from App-level scene assets and sends a one-off `composition_config` with video creation jobs. The backend validates the config and post-processes generated frames over an image background before writing the video.

**Tech Stack:** React + TypeScript frontend, FastAPI multipart route, Python video creation service, OpenCV/NumPy/Pillow-style frame processing.

## Global Constraints

- Video Creation composition adjustments only affect the current generation job.
- Do not write changes back to Scene Assets in this version.
- If no avatar scene/background exists, keep current Video Creation behavior.
- Reject video backgrounds for Video Creation in this first version.
- Local runtime is unavailable; commit locally, sync via git bundle, and run verification on `8.92.9.220:/home/ly/opentalking`.

---

### Task 1: Backend Composition Config Parsing

**Files:**
- Modify: `apps/api/routes/video_creation.py`
- Test: `apps/api/tests/test_video_creation.py`

**Interfaces:**
- Produces: `_parse_video_composition_config(raw: str | None) -> dict[str, object] | None`
- Produces: optional `composition_config` argument passed to `VideoCreationService.create_from_audio_file`, `create_from_tts_text`, and `create_reference_video`

- [ ] **Step 1: Write failing route tests**

Add tests that post `composition_config` JSON to `/video-creation/jobs` and assert the fake service receives a dict. Add an invalid JSON test expecting HTTP 400.

- [ ] **Step 2: Run route tests to verify they fail**

Run on server after sync: `uv run pytest apps/api/tests/test_video_creation.py -k "composition_config" -q`

- [ ] **Step 3: Implement parser and route forwarding**

Add a FastAPI `Form(default=None)` field named `composition_config`, parse JSON, require a JSON object, and pass the parsed dict to service calls.

- [ ] **Step 4: Run route tests to verify they pass**

Run on server after sync: `uv run pytest apps/api/tests/test_video_creation.py -k "composition_config" -q`

### Task 2: Backend Frame Compositing

**Files:**
- Modify: `opentalking/video_creation.py`
- Test: `apps/api/tests/test_video_creation.py`

**Interfaces:**
- Consumes: `composition_config: Mapping[str, object] | None`
- Produces: `_normalize_video_composition_config(settings: object, config: Mapping[str, object] | None) -> dict[str, object] | None`
- Produces: `_apply_video_composition(frames: list[np.ndarray], *, config: Mapping[str, object] | None) -> list[np.ndarray]`

- [ ] **Step 1: Write failing service tests**

Add a test that creates a temporary scene background image, sends composition config to `VideoCreationService.create_from_audio_file`, monkeypatches `_write_video_only`, and asserts written frames contain the background color behind a transparent RGBA generated frame.

- [ ] **Step 2: Run service test to verify it fails**

Run on server after sync: `uv run pytest apps/api/tests/test_video_creation.py -k "composites_generated_frames" -q`

- [ ] **Step 3: Implement minimal composition**

In `_create_from_pcm`, normalize `composition_config` before rendering, apply it before `_write_video_only`, support image backgrounds only, and raise `ValueError("video backgrounds are not supported for video creation")` for video backgrounds.

- [ ] **Step 4: Run service tests to verify they pass**

Run on server after sync: `uv run pytest apps/api/tests/test_video_creation.py -k "composition" -q`

### Task 3: Frontend API and App Data Flow

**Files:**
- Modify: `apps/web/src/lib/api.ts`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/components/VideoCreationWorkspace.tsx`
- Test: `tests/unit/test_local_audio_frontend.py`

**Interfaces:**
- Produces: `VideoCreationCompositionConfig` TypeScript type
- Produces: `compositionConfig?: VideoCreationCompositionConfig | null` on `CreateVideoCreationJobInput`
- Consumes: `sceneBackgrounds`, `sceneCompositions`, `selectedSceneIdsByAvatar` props in `VideoCreationWorkspace`

- [ ] **Step 1: Write failing frontend text tests**

Add assertions that `VideoCreationWorkspace` receives scene props from `App.tsx`, defines `compositionConfig`, renders `生成前预览`, and appends `composition_config` in `createVideoCreationJob`.

- [ ] **Step 2: Run frontend text tests to verify they fail**

Run on server after sync: `uv run pytest tests/unit/test_local_audio_frontend.py -k "video_creation" -q`

- [ ] **Step 3: Implement API and prop plumbing**

Add the TypeScript composition type, JSON form field, and pass App scene state into `VideoCreationWorkspace`.

- [ ] **Step 4: Run frontend text tests to verify they pass**

Run on server after sync: `uv run pytest tests/unit/test_local_audio_frontend.py -k "video_creation" -q`

### Task 4: Frontend Preview and Controls

**Files:**
- Modify: `apps/web/src/components/VideoCreationWorkspace.tsx`
- Test: `tests/unit/test_local_audio_frontend.py`

**Interfaces:**
- Consumes: scene props and `buildApiUrl`
- Produces: one-off local state `{ backgroundId, backgroundColor, avatarFit, avatarAnchor, avatarScale, avatarOffsetX, avatarOffsetY }`

- [ ] **Step 1: Write failing UI string tests**

Assert `VideoCreationWorkspace.tsx` contains `生成前预览`, `本次生成`, `水平位置`, `垂直位置`, `人物缩放`, and `compositionConfig`.

- [ ] **Step 2: Run tests to verify they fail**

Run on server after sync: `uv run pytest tests/unit/test_local_audio_frontend.py -k "video_creation_composition" -q`

- [ ] **Step 3: Implement preview and controls**

Show the selected background image when present, show the selected avatar preview above it, provide sliders for X/Y/scale, reset controls, and build `compositionConfig` for generation. Keep fallback copy for no background.

- [ ] **Step 4: Run tests to verify they pass**

Run on server after sync: `uv run pytest tests/unit/test_local_audio_frontend.py -k "video_creation_composition" -q`

### Task 5: Server Sync and Verification

**Files:**
- No code files; sync and run commands.

**Interfaces:**
- Consumes: local commits on `codex/video-creation-composition`
- Produces: matching server git history and verification output

- [ ] **Step 1: Commit local changes**

Run: `git add docs apps opentalking tests && git commit -m "Add video creation scene composition"`

- [ ] **Step 2: Create and upload git bundle**

Run locally: `git bundle create /tmp/video-creation-composition.bundle HEAD`

Upload to server: `sshpass -p 'ly.123' scp /tmp/video-creation-composition.bundle ly@8.92.9.220:/tmp/video-creation-composition.bundle`

- [ ] **Step 3: Apply bundle on server**

Run on server: `cd /home/ly/opentalking && git fetch /tmp/video-creation-composition.bundle codex/video-creation-composition:codex/video-creation-composition && git switch codex/video-creation-composition`

- [ ] **Step 4: Run verification**

Run on server:

```bash
cd /home/ly/opentalking
uv run mypy opentalking/core opentalking/events opentalking/avatar apps/api apps/unified apps/cli --ignore-missing-imports
uv run pytest apps/api/tests/test_video_creation.py tests/unit/test_local_audio_frontend.py -q
```

- [ ] **Step 5: Start service for manual review**

Run on server:

```bash
cd /home/ly/opentalking
bash scripts/quickstart/stop_all.sh || true
bash scripts/start_unified.sh --mock --api-port 8211 --web-port 5281
```
