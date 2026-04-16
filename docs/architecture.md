# 架构

## 组件

| 组件 | 职责 |
|------|------|
| **apps/web** | 控制台：创建会话、WebRTC 收流、SSE 字幕与状态 |
| **apps/api** | REST：会话、Avatar 列表、转发 WebRTC SDP 到 Worker |
| **apps/worker** | 消费 Redis 任务：`init` / `speak` / `interrupt`；TTS + 模型适配器 + WebRTC 推流 |
| **Redis** | 任务队列 `opentalking:task_queue`；事件频道 `opentalking:events:{session_id}` |
| **packages/** | 可复用库：核心协议、模型/TTS/RTC 适配器、Avatar 校验 |

**单进程模式**（`apps/unified`）：使用 `InMemoryRedis`（内存哈希 + 异步任务队列 + 简易 pub/sub）替代 Redis；`SessionRunner` 仍在后台任务中消费同一队列，WebRTC 应答由 API 在同进程内调用 `session_runners` 完成，无需 `OPENTALKING_WORKER_URL`。

## 数据流（简述）

1. 客户端 `POST /sessions` → API 写 Redis 会话哈希并 `RPUSH` `init` 任务。
2. Worker `BRPOP` 加载模型与 Avatar，准备 `WebRTCSession`。
3. 客户端生成本地 SDP Offer → `POST /sessions/{id}/webrtc/offer` → API 转发 Worker → 返回 Answer。
4. `POST /sessions/{id}/speak` → Worker TTS 流式 PCM → 模型适配器生成视频帧 → 入 WebRTC 队列；并行 `PUBLISH` 事件供 SSE。

详见 [render-pipeline.md](render-pipeline.md)。
