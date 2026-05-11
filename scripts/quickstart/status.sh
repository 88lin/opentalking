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

export DIGITAL_HUMAN_HOME="${DIGITAL_HUMAN_HOME:-$default_home}"

api_port="${OPENTALKING_API_PORT:-8000}"
web_port="${OPENTALKING_WEB_PORT:-5173}"
omnirt_url="${OMNIRT_ENDPOINT:-http://127.0.0.1:9000}"
run_dir="$DIGITAL_HUMAN_HOME/run"

show_pid() {
  local name="$1"
  local pid_file="$2"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name: running pid=$pid"
      return
    fi
  fi
  echo "$name: not running from quickstart pid file"
}

check_url() {
  local label="$1"
  local url="$2"
  if curl --max-time 2 -fsS "$url" >/dev/null 2>&1; then
    echo "$label: ok ($url)"
  else
    echo "$label: unavailable ($url)"
  fi
}

echo "DIGITAL_HUMAN_HOME=$DIGITAL_HUMAN_HOME"
show_pid "OpenTalking API" "$run_dir/opentalking-api.pid"
show_pid "OpenTalking frontend" "$run_dir/opentalking-web.pid"
show_pid "OmniRT Wav2Lip" "$run_dir/omnirt-wav2lip.pid"
show_pid "OmniRT FlashTalk endpoint" "$run_dir/omnirt-flashtalk.pid"
check_url "OpenTalking API /models" "http://127.0.0.1:$api_port/models"
check_url "OpenTalking frontend" "http://127.0.0.1:$web_port"
check_url "OmniRT /v1/audio2video/models" "$omnirt_url/v1/audio2video/models"
