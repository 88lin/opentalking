# Quickstart

> 中文 · [English](./quickstart.en.md)

最快的方式：3 行命令一键部署。

```bash
git clone https://github.com/<org>/opentalking && cd opentalking
cp .env.example .env
bash scripts/install.sh
```

打开 http://localhost:5173 即可对话。

## 一键脚本做了什么

`scripts/install.sh` 流程：

1. `scripts/detect_hardware.sh` 探测硬件（CUDA / Ascend / CPU）→ 选 profile
2. 根据 profile 选择 `deploy/compose/docker-compose.<profile>.yml`
3. `docker compose pull` 拉镜像
4. `scripts/up.sh` 起容器（redis + omnirt + api + worker + web）
5. `scripts/ensure_omnirt.sh` 等 omnirt 健康检查通过
6. 输出访问地址

## Native 模式（不用 Docker）

```bash
bash scripts/install.sh native
```

需要自行启动 omnirt，并在 `.env` 中配置 `OMNIRT_ENDPOINT`。

## 前端联调（无 GPU）

```bash
docker compose -f deploy/compose/docker-compose.dev.yml up
```

此模式不连 omnirt，API 走 `OPENTALKING_INFERENCE_MOCK=1`，仅供前端样式调试。

## 自定义环境变量

模板见 [.env.example](../.env.example)，分组：

- `Service` — 端口 / profile
- `Inference (omnirt)` — endpoint / API key
- `Storage` — avatar / voice 目录、Redis URL
- `STT / TTS / LLM` — 各 provider 凭据

详细每项含义见 [configuration.md](configuration.md)。

## 常见问题

- **Docker 镜像拉不下来**：检查 `OMNIRT_ENDPOINT` 是否配置；本地 omnirt 镜像源参考 [datascale-ai/omnirt](https://github.com/datascale-ai/omnirt)。
- **omnirt 启动慢**：首次启动需下载模型权重，可达几十 GB；进度看 `docker logs <omnirt-container>`。
- **FlashTalk 14B 权重大**：`bash scripts/download_flashtalk.sh` 可单独预拉，否则 omnirt 首次调用时按需拉取。

更多硬件适配见 [hardware.md](hardware.md)，部署变体见 [deployment.md](deployment.md)。
