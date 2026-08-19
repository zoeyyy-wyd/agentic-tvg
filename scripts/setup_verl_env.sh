#!/usr/bin/env bash
# Agentic-TVG: one-shot installer for the `verl` conda env.
# Target: A100 80GB (sm80), driver 610.43.02 / CUDA UMD 13.3, Ubuntu 22.04.
# Rationale for every pin below: see ENVIRONMENT.md
set -euo pipefail

ENV_NAME="${ENV_NAME:-verl}"
PY_VER="3.12"
CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"
VERL_SRC="${VERL_SRC:-$HOME/src/verl}"

TORCH_VER="2.11.0"
TV_VER="0.26.0"
TA_VER="2.11.0"
CUDA_TAG="cu130"
VLLM_VER="0.24.0"
TRANSFORMERS_VER="5.10.4"   # actually pinned in requirements.txt
FA_VER="2.8.3"
FA_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-${FA_VER}%2B${CUDA_TAG}torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }

source "$CONDA_SH"

# ---------------------------------------------------------------- 0. env
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  log "env '${ENV_NAME}' already exists. To reinstall from scratch (e.g. it is an empty stub), run: conda env remove -n ${ENV_NAME}"
  read -r -p "Continue installing into the existing env? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
else
  log "Creating conda env ${ENV_NAME} (python ${PY_VER})"
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"
python -V

log "Installing ffmpeg (backend for PyAV / torchcodec)"
conda install -y -c conda-forge ffmpeg

# ---------------------------------------------------------------- 1. torch (must come first)
log "Installing torch ${TORCH_VER}+${CUDA_TAG}"
pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}"

python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| avail:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU, aborting"
PY

# ---------------------------------------------------------------- 2. vllm
log "Installing vllm ${VLLM_VER}"
pip install "vllm==${VLLM_VER}"

python - <<'PY'
import torch
assert torch.__version__.startswith("2.11.0"), f"vllm replaced torch: {torch.__version__}"
assert torch.version.cuda.startswith("13"), f"unexpected CUDA variant: {torch.version.cuda}"
print("torch survived:", torch.__version__)
PY

# ---------------------------------------------------------------- 3. flash-attn (FA2 is the only option on sm80)
log "Installing flash-attn ${FA_VER} (prebuilt wheel first)"
if ! pip install "$FA_WHEEL"; then
  log "Prebuilt wheel failed, falling back to a source build (40-90 min, MAX_JOBS=32)"
  pip install packaging ninja
  MAX_JOBS=32 pip install "flash-attn==${FA_VER}" --no-build-isolation
fi

# ---------------------------------------------------------------- 4. verl runtime dependencies
log "Installing verl runtime dependencies (requirements.txt)"
pip install -r "$(dirname "$0")/../requirements.txt"

# ---------------------------------------------------------------- 5. verl itself
log "Installing verl v0.9.0 (from source, --no-deps)"
if [[ ! -d "$VERL_SRC" ]]; then
  mkdir -p "$(dirname "$VERL_SRC")"
  git clone --branch v0.9.0 --depth 1 https://github.com/volcengine/verl.git "$VERL_SRC"
fi
pip install --no-deps -e "$VERL_SRC"

# ---------------------------------------------------------------- 6. verification
log "Verifying"
python "$(dirname "$0")/check_env.py"

log "Done. Activate with: conda activate ${ENV_NAME}"
