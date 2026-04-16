# 贡献指南

1. **接口优先**：新增模型/TTS/RTC 实现时，优先满足 `packages/core` 中的 Protocol，避免在 `apps/` 内写死具体类名。
2. **小步提交**：适配器、API 路由、Worker 行为尽量分提交，便于审查。
3. **文档**：用户可见行为变更请同步 `README.md` 或 `docs/`。
4. **测试**：为新工具函数或解析逻辑补充 `pytest`；端到端 WebRTC 需本机 Redis + 浏览器，可手工验证。

欢迎通过 Issue / PR 讨论适配器边界与事件载荷格式。
