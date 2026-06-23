# Video Creation Composition Design

## Goal

Bring scene backgrounds and avatar placement into the Video Creation workflow so offline generated videos can match the visual composition users preview before generation.

## Product Decision

Video Creation uses a pre-generation composition preview, not an immersive conversation mode. The page is an offline production workspace: users choose an avatar, script or audio, model settings, and then confirm the final frame composition before clicking generate.

Composition adjustments in Video Creation are one-off job settings. They must not update the active realtime conversation scene or mutate Scene Asset compositions unless the user explicitly uses a future save action.

## User Experience

When a user selects an avatar on the Video Creation page:

- If the avatar has an active scene composition, the preview uses that scene's background, background color, avatar fit, anchor, and scale.
- If the avatar has no active scene composition, Video Creation keeps the existing default avatar-only behavior.
- The user can adjust horizontal position, vertical position, and scale for this generation.
- The preview shows background plus avatar in an output-frame container before generation.
- Reset returns the one-off transform to `x: 0`, `y: 0`, `scale: 1`.

The first version does not add a write-back action to Scene Assets. It only sends composition data with the video creation job.

## Frontend Architecture

`App.tsx` already owns scene backgrounds, scene compositions, and selected scene ids by avatar. Video Creation should receive those values as props and derive the current avatar's active scene with the same rule as realtime conversation:

1. Use `selectedSceneIdsByAvatar[avatarId]` if it points to a scene for the selected avatar.
2. Otherwise use the first composition whose `avatar_id` matches the selected avatar.
3. Otherwise use no scene.

`VideoCreationWorkspace.tsx` adds:

- Props for `sceneBackgrounds`, `sceneCompositions`, and `selectedSceneIdsByAvatar`.
- Local state for `videoAvatarAdjust: { x: number; y: number; scale: number }`.
- A composition preview panel using the selected scene and background data.
- Background selection for this job only.
- X, Y, and scale controls for this job only.
- A `videoComposition` payload passed to `createVideoCreationJob`.

The UI stays work-focused and data-dense. The preview is prominent, but controls remain visible because this is a production page rather than an immersive live demo page.

## API Contract

`createVideoCreationJob` accepts an optional multipart field named `composition_config`. The field is JSON:

```json
{
  "scene_composition_id": "scene-example",
  "background_id": "bg-example",
  "background_color": "#ffffff",
  "avatar_fit": "contain",
  "avatar_anchor": "center",
  "avatar_scale": 1.1,
  "avatar_offset_x": 80,
  "avatar_offset_y": -24
}
```

All fields are optional except that a useful composition must include at least `background_id` or a non-empty scene-derived background color. If the field is absent, current video generation behavior remains unchanged.

Validation rules:

- `background_id` must resolve under the configured scene assets directory when present.
- `avatar_fit` must be `contain` or `cover`.
- `avatar_anchor` must be `center`, `bottom`, `left`, or `right`.
- `avatar_scale` must be between `0.1` and `4.0`.
- Offsets are pixel values and must be between `-2000` and `2000`.
- Video backgrounds are not composited in the first version; the backend rejects a video background for Video Creation with a clear message.

## Backend Architecture

`VideoCreationService` accepts `composition_config` on audio upload, TTS text, and reference video generation.

The first version applies composition after model frame generation and before writing `video_only.mp4`:

1. Generate avatar frames as today.
2. If no composition config or no image background is provided, write frames unchanged.
3. If an image background is provided, resize/crop it to the generated frame size.
4. Place each generated frame over the background according to fit, anchor, base scale, and one-off offsets.
5. Use alpha blending if generated frames contain an alpha channel; otherwise paste the RGB frame as an opaque layer.
6. Continue muxing audio and export metadata as today.

This keeps model-specific audio-to-video logic untouched and makes composition a reusable post-processing step.

## Testing

Backend tests cover:

- API route parses `composition_config` and passes it to `VideoCreationService`.
- Invalid JSON is rejected with HTTP 400.
- Service rejects unknown or video backgrounds for Video Creation.
- Service composites generated RGBA frames over an image background with scale and offsets.
- Existing jobs without composition still behave as before.

Frontend text-level tests cover:

- Video Creation receives scene data from `App.tsx`.
- `VideoCreationWorkspace.tsx` exposes the composition preview and controls.
- `createVideoCreationJob` sends `composition_config`.

## Non-Goals

- No immersive mode on Video Creation in this first version.
- No automatic write-back to Scene Assets.
- No video-background compositing in Video Creation yet.
- No subtitle rendering into generated videos in this first version.
