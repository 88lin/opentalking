# 本地开发

## 单进程（无 Redis）

```bash
./run.sh
# 或: opentalking-unified --port 8000
```

与 `apps/web` 的 Vite 代理（默认转发到 `localhost:8000`）对齐。

## 多进程（API + Worker + Redis）

1. `./scripts/setup.sh` 或手动 `pip install -e ".[dev]"`。
2. 启动 Redis（本机或 Docker）。
3. 终端 A：`opentalking-worker`（设置 `OPENTALKING_REDIS_URL`、`OPENTALKING_AVATARS_DIR`、`OPENTALKING_TORCH_DEVICE`）。
4. 终端 B：`opentalking-api`（设置 `OPENTALKING_WORKER_URL` 指向 Worker 信令地址，默认 `http://127.0.0.1:9001`）。
5. `cd apps/web && npm run dev`，通过 Vite 代理访问 API（`/api` → `localhost:8000`）。

## 环境变量摘要

见仓库根目录 `.env.example`。

## 测试

```bash
pytest apps/api/tests -q
```
