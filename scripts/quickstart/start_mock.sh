#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

env_file="${OPENTALKING_QUICKSTART_ENV:-$script_dir/env}"
if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1090
  source "$env_file"
fi

echo "Starting OpenTalking mock/self-test mode"
echo "This clears OmniRT endpoint variables for the OpenTalking API process."

bash "$script_dir/start_opentalking.sh" --mock
bash "$script_dir/start_frontend.sh"

echo ""
echo "Open the app:"
echo "  http://127.0.0.1:${OPENTALKING_WEB_PORT:-5173}"
echo ""
echo "Select mock / 无驱动模式 to test without a real driver model."
