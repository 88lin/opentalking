#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OPENTALKING_UNIFIED_HOST="${OPENTALKING_UNIFIED_HOST:-0.0.0.0}"
export OPENTALKING_UNIFIED_PORT="${OPENTALKING_UNIFIED_PORT:-8000}"
export OPENTALKING_AVATARS_DIR="${OPENTALKING_AVATARS_DIR:-./examples/avatars}"

exec opentalking-unified "$@"
