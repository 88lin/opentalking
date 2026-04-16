#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-ascend}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

cd "$PROJECT_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -i "$PIP_INDEX_URL" -e ".[engine,ascend,demo]"

export HF_ENDPOINT
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cat <<EOF
Ascend deployment prepared.

Next run:
  cd $PROJECT_DIR
  source $VENV_DIR/bin/activate
  torchrun --nproc_per_node=\${OPENTALKING_FLASHTALK_GPU_COUNT:-8} -m opentalking.server \\
    --ckpt_dir \${OPENTALKING_FLASHTALK_CKPT_DIR:-./models/SoulX-FlashTalk-14B} \\
    --wav2vec_dir \${OPENTALKING_FLASHTALK_WAV2VEC_DIR:-./models/chinese-wav2vec2-base} \\
    --port \${OPENTALKING_FLASHTALK_PORT:-8765}
EOF
