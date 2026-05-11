#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
default_home="$(cd -- "$repo_root/.." && pwd)"

env_file="${OPENTALKING_QUICKSTART_ENV:-$script_dir/env}"
if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1090
  source "$env_file"
fi

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/quickstart/start_opentalking.sh [--mock] [--omnirt URL]

Options:
  --mock        Force mock/self-test mode by clearing OmniRT endpoint variables.
  --omnirt URL Set OMNIRT_ENDPOINT for this process, for example http://127.0.0.1:9000.
  --help       Show this help.
USAGE
}

mock_mode=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock)
      mock_mode=1
      shift
      ;;
    --omnirt)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --omnirt" >&2
        exit 2
      fi
      export OMNIRT_ENDPOINT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export DIGITAL_HUMAN_HOME="${DIGITAL_HUMAN_HOME:-$default_home}"
export OMNIRT_MODEL_ROOT="${OMNIRT_MODEL_ROOT:-$DIGITAL_HUMAN_HOME/models}"

if [[ "$mock_mode" == "1" ]]; then
  export OMNIRT_ENDPOINT=""
  export OPENTALKING_OMNIRT_ENDPOINT=""
fi

api_port="${OPENTALKING_API_PORT:-8000}"
run_dir="$DIGITAL_HUMAN_HOME/run"
log_dir="$DIGITAL_HUMAN_HOME/logs"
pid_file="$run_dir/opentalking-api.pid"
log_file="$log_dir/opentalking-api.log"

mkdir -p "$run_dir" "$log_dir"

if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "OpenTalking API is already running: pid=$old_pid port=$api_port"
    echo "Log: $log_file"
    exit 0
  fi
  rm -f "$pid_file"
fi

if [[ ! -f "$repo_root/.venv/bin/activate" ]]; then
  echo "Missing virtualenv: $repo_root/.venv" >&2
  echo "Run this first: cd \"$repo_root\" && uv sync && uv pip install -e \".[dev]\"" >&2
  exit 1
fi

if [[ ! -f "$repo_root/.env" && -f "$repo_root/.env.example" ]]; then
  cp "$repo_root/.env.example" "$repo_root/.env"
  echo "Created $repo_root/.env from .env.example. Edit it for LLM/STT credentials if needed."
fi

echo "Starting OpenTalking API"
echo "  repo:    $repo_root"
echo "  home:    $DIGITAL_HUMAN_HOME"
echo "  port:    $api_port"
echo "  log:     $log_file"
if [[ -n "${OMNIRT_ENDPOINT:-}" ]]; then
  echo "  omnirt:  $OMNIRT_ENDPOINT"
else
  echo "  omnirt:  disabled or read from .env"
fi

(
  cd "$repo_root"
  source .venv/bin/activate
  exec opentalking-unified
) >"$log_file" 2>&1 &

pid="$!"
echo "$pid" > "$pid_file"

for _ in {1..60}; do
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "OpenTalking API exited during startup. Last log lines:" >&2
    tail -80 "$log_file" >&2 || true
    rm -f "$pid_file"
    exit 1
  fi
  if curl --max-time 2 -fsS "http://127.0.0.1:$api_port/models" >/dev/null 2>&1; then
    echo "OpenTalking API is up: http://127.0.0.1:$api_port"
    exit 0
  fi
  sleep 1
done

echo "OpenTalking API did not become ready in 60s. Last log lines:" >&2
tail -80 "$log_file" >&2 || true
exit 1
