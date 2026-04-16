# Avatar 资产格式

## 目录布局

每个 Avatar 一个子目录，**必须**包含 `manifest.json`。

- **wav2lip**：`frames/` 下若干 `.png` / `.jpg`（按文件名排序）。
- **musetalk**：`full_frames/` 下同样为有序图像序列（完整帧；后续可扩展 mask、latent 等子目录）。

推荐提供 `preview.png` 供前端展示。

## manifest.json 字段

| 字段 | 说明 |
|------|------|
| `id` | 唯一 ID |
| `name` | 展示名（可选） |
| `model_type` | `wav2lip` 或 `musetalk` |
| `fps` | 目标帧率 |
| `sample_rate` | 音频采样率（与 TTS 输出对齐，常用 16000） |
| `width` / `height` | 视频分辨率 |
| `version` | 资产版本字符串 |
| `metadata` | 任意附加信息 |

校验逻辑见 `opentalking.avatars.validator`。
