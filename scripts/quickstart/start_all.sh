#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

env_file="${OPENTALKING_QUICKSTART_ENV:-$script_dir/env}"
if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1090
  source "$env_file"
fi

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/quickstart/start_all.sh [--mock] [--omnirt URL]

Examples:
  bash scripts/quickstart/start_all.sh --mock
  bash scripts/quickstart/start_all.sh --omnirt http://127.0.0.1:9000

This starts OpenTalking API and frontend. It does not start OmniRT itself.
Start OmniRT Wav2Lip or FlashTalk first when using a real driver model.
USAGE
}

mode_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock)
      mode_args+=(--mock)
      shift
      ;;
    --omnirt)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --omnirt" >&2
        exit 2
      fi
      mode_args+=(--omnirt "$2")
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

bash "$script_dir/start_opentalking.sh" "${mode_args[@]}"
bash "$script_dir/start_frontend.sh"

echo ""
echo "Open the app:"
echo "  http://127.0.0.1:${OPENTALKING_WEB_PORT:-5173}"
