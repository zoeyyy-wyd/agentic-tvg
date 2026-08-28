#!/usr/bin/env bash
# QA pipeline: empty checkout -> sft_{train,val}.parquet (README "Run").
#
#   bash prepare_data.sh              # ~36G downloads + render, 30-60 min
#   SKIP_ARMB=0 bash prepare_data.sh  # also fetch Arm-B image-CoT (4.7G; ablations were cut 2026-08-26)
#
# Re-run safe: hf download resumes, zips are re-fetched only if missing,
# render overwrites its parquet output.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)
source "${REPO}/env_setup/preflight.sh"

SKIP_ARMB=${SKIP_ARMB:-1}   # ablations cut; images only on explicit request
mkdir -p logs data/archives data/videos/selfqa data/videos/rl_val data/videos/geminicot data/images models
step() { echo -e "\n\033[1;32m==> $*\033[0m"; }

avail=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$avail" -ge 70 ] || { echo "need >=70G free (36G data + ~30G ckpts), have ${avail}G" >&2; exit 1; }

# ---------------------------------------------------------------- 1. downloads
# NOTE: zip names MUST be positional args. `--include a.zip b.zip` makes the
# second name positional and silently discards the --include pattern (verified
# the hard way 2026-08-25).
step "Downloading in parallel (logs/dl_*.log)"
hf download longvideotool/LongVT-Parquet --repo-type dataset \
    --local-dir data/annotations > logs/dl_anno.log 2>&1 &
p_anno=$!
hf download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir models/Qwen3-VL-4B-Instruct > logs/dl_model.log 2>&1 &
p_model=$!
hf download longvideotool/LongVT-Source selfqa_1.zip rl_val_1.zip \
    --repo-type dataset --local-dir data/archives > logs/dl_selfqa.log 2>&1 &
p_selfqa=$!
hf download longvideotool/LongVT-Source geminicot_1.zip geminicot_2.zip \
    --repo-type dataset --local-dir data/archives > logs/dl_gemini.log 2>&1 &
p_gemini=$!
p_armb=""
if [ "$SKIP_ARMB" != "1" ]; then
    hf download longvideotool/LongVT-Source llavacot_1.zip openvlthinker_1.zip wemath_1.zip \
        --repo-type dataset --local-dir data/archives > logs/dl_armb.log 2>&1 &
    p_armb=$!
fi

# Report every failure, not just the first.
rc=0
for job in "annotations:$p_anno" "model:$p_model" "selfqa+rl_val:$p_selfqa" \
           "geminicot:$p_gemini" ${p_armb:+"arm-b-images:$p_armb"}; do
    name=${job%%:*}; pid=${job##*:}
    if wait "$pid"; then echo "  [ok]   $name"; else echo "  [FAIL] $name -- see logs/"; rc=1; fi
done
[ "$rc" -eq 0 ] || { echo "download(s) failed, nothing else ran" >&2; exit 1; }

# ---------------------------------------------------------------- 2. extract
step "Extracting (zips removed after extraction to cap peak disk)"
x() { [ -e "$1" ] && { unzip -q -o "$1" -d "$2" && rm -f "$1"; } || true; }
x data/archives/selfqa_1.zip     data/videos/selfqa
x data/archives/rl_val_1.zip     data/videos/rl_val
for z in data/archives/geminicot_*.zip; do x "$z" data/videos/geminicot; done
if [ "$SKIP_ARMB" != "1" ]; then
    for z in data/archives/llavacot_1.zip data/archives/openvlthinker_1.zip data/archives/wemath_1.zip; do
        x "$z" data/images
    done
fi
du -sh data/videos/* data/images models data/annotations 2>/dev/null || true

# ---------------------------------------------------------------- 3. render
step "Allocation plan (CHECK: joined 1157 | SFT 600q -> 1379 traces + geminicot 600 | RL 893)"
python data_prep/render_traces.py --plan-only

step "Full render, decodes ~32K frames (CHECK: rendered ~1.9K+, small drop counters)"
python data_prep/render_traces.py

step "Done. Next: QA SFT smoke -- README, Run section"
