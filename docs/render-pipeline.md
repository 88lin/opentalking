# 渲染管线

## Worker 内路径

1. **TTS**：`EdgeTTSAdapter.synthesize_stream` 将 `edge-tts` 返回的 MP3 片段持续写入 `ffmpeg` stdin，边解码边产出 16-bit PCM，并按 `chunk_ms`（默认 20ms）切片为 `AudioChunk`。
2. **特征**：各模型适配器的 `extract_features`（当前为 RMS + 帧数估算占位，可替换为 Whisper/Mel）。
3. **推理**：`infer` 返回与音频 chunk 对齐的预测列表；无神经网络权重时为 `None` 列表，走帧合成回退。
4. **合成**：`compose_frame` 从 Avatar 帧序列取图并输出 `VideoFrameData`（BGR uint8）。
5. **推流**：`WebRTCSession` 将帧送入 `VideoStreamTrack`，PCM 送入 `AudioStreamTrack`。

## 空闲循环

非播报时 `_idle_loop` 按 Avatar `fps` 调用 `idle_frame`，避免 WebRTC 视频轨阻塞。

## 打断

`interrupt` 设置 `_interrupt`；当前轮 `speak` 在 TTS 迭代中检测后提前结束。
