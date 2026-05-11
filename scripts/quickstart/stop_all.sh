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

run_dir="$DIGITAL_HUMAN_HOME/run"

stop_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name: not started by quickstart scripts"
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "$name: stale pid file removed"
    rm -f "$pid_file"
    return
  fi

  echo "Stopping $name: pid=$pid"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$pid_file"
      echo "$name: stopped"
      return
    fi
    sleep 0.5
  done

  echo "$name: still running, sending SIGKILL"
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pid_file"
}

stop_pid_file "OpenTalking API" "$run_dir/opentalking-api.pid"
stop_pid_file "OpenTalking frontend" "$run_dir/opentalking-web.pid"
stop_pid_file "OmniRT Wav2Lip" "$run_dir/omnirt-wav2lip.pid"
stop_pid_file "OmniRT FlashTalk endpoint" "$run_dir/omnirt-flashtalk.pid"
