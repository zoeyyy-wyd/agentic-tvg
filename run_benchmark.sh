#!/usr/bin/env bash
# VideoSIAH-Eval (652 open-ended QA, 244 videos, ~109G) through the unified
# evaluator (run_grpo.sh trainer.val_only=True) -- acc via the judge; there is
# no GT time window in this benchmark, so evidence_iou is identically 0.
#
# The whole video set does not fit on this disk (~27G free vs 109G), so the
# benchmark runs in zip-sized chunks: download one ~10G zip -> extract ->
# build the chunk parquet (rows whose video is now local) -> val_only pass ->
# keep the rollout jsonl -> delete the videos. Each chunk pays a fresh vLLM
# spin-up; 12 restarts is the price of streaming 109G through 27G.
#
# Resume-safe: a chunk whose jsonl already exists under results/bench-<name>/
# is skipped, so re-running after a crash continues where it left off.
#
# Usage:
#   MODEL_PATH=results/rft/merged BENCH_NAME=rft bash run_benchmark.sh        # all 12 chunks
#   MODEL_PATH=... BENCH_NAME=grpo bash run_benchmark.sh 3 7                  # only chunks 3 and 7
# Then:
#   python data_prep/score_rollouts.py results/bench-rft/chunk_*.jsonl
#
# Numbers are comparable across OUR stages (same judge, same harness), not to
# the LongVT paper's table (different judge model and eval framework).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)
source "${REPO}/env_setup/preflight.sh"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH=... (e.g. results/rft/merged)}
BENCH_NAME=${BENCH_NAME:?set BENCH_NAME= short tag, e.g. rft / grpo / base}
BENCH_NAME=${BENCH_NAME//_/-}   # run_grpo.sh maps EXP_NAME _ -> - in RESULT_NAME; pre-sanitize so our paths agree
CHUNKS=("$@"); [ ${#CHUNKS[@]} -gt 0 ] || CHUNKS=(1 2 3 4 5 6 7 8 9 10 11 12)

# GPU courtesy check (protocol: /tmp/gpu-owner.lock). IGNORE_LOCK=1 overrides
# a stale entry; never use it to preempt a live run.
if [ -f /tmp/gpu-owner.lock ] && ! grep -q '^free' /tmp/gpu-owner.lock \
   && [ "${IGNORE_LOCK:-0}" != "1" ]; then
    echo "[abort] GPU lock not free: $(head -c 200 /tmp/gpu-owner.lock)" >&2
    exit 1
fi

QA_PARQUET=data/annotations/videosiah_eval/data/test-00000-of-00001.parquet
[ -f "${QA_PARQUET}" ] || hf download longvideotool/VideoSIAH-Eval data/test-00000-of-00001.parquet \
    --repo-type dataset --local-dir data/annotations/videosiah_eval

VIDEO_DIR=data/videos/videosiah_eval
OUT_DIR=results/bench-${BENCH_NAME}
mkdir -p "${VIDEO_DIR}" data/archives "${OUT_DIR}"

for i in "${CHUNKS[@]}"; do
    out_jsonl=${OUT_DIR}/chunk_${i}.jsonl
    if [ -s "${out_jsonl}" ]; then
        echo "[skip] chunk ${i} -- ${out_jsonl} already exists"
        continue
    fi

    # Peak per chunk = zip (~10G) + extracted videos (~11G) before the zip is
    # deleted; 24G keeps a few G of margin for logs/tb on top of that.
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    [ "${free_gb}" -ge 24 ] || {
        echo "[abort] ${free_gb}G free; a chunk needs ~24G (zip + extracted, side by side)" >&2
        exit 1; }

    echo "==> chunk ${i}/12: download + extract"
    # zip name MUST be a positional arg -- --include silently no-ops here
    # (same gotcha as prepare_data.sh, learned 2026-08-25).
    hf download longvideotool/VideoSIAH-Eval "videosiaheval_${i}.zip" \
        --repo-type dataset --local-dir data/archives
    unzip -q -o "data/archives/videosiaheval_${i}.zip" -d "${VIDEO_DIR}"
    rm -f "data/archives/videosiaheval_${i}.zip"

    echo "==> chunk ${i}/12: build parquet"
    python data_prep/prepare_videosiah_eval.py --videos-dir "${VIDEO_DIR}" \
        --out data/processed/videosiah_chunk.parquet
    rows=$(python -c "import pyarrow.parquet as pq,sys; print(pq.read_metadata(sys.argv[1]).num_rows)" \
           data/processed/videosiah_chunk.parquet)
    if [ "${rows}" -eq 0 ]; then
        echo "[warn] chunk ${i} matched 0 QA rows; skipping eval" >&2
        rm -rf "${VIDEO_DIR:?}"/*
        continue
    fi

    echo "==> chunk ${i}/12: val_only over ${rows} rows"
    VAL_FILE="${REPO}/data/processed/videosiah_chunk.parquet" \
    MODEL_PATH="${MODEL_PATH}" EXP_NAME="bench_${BENCH_NAME}_${i}" \
        bash run_grpo.sh trainer.val_only=True

    src=results/bench-${BENCH_NAME}-${i}/val_rollouts/0.jsonl
    [ -s "${src}" ] || { echo "[abort] rollout dump missing: ${src}" >&2; exit 1; }
    cp "${src}" "${out_jsonl}"

    rm -rf "${VIDEO_DIR:?}"/*   # ~10G back before the next chunk
    echo "==> chunk ${i}/12 done -> ${out_jsonl} (${rows} rows)"
done

echo
python data_prep/score_rollouts.py "${OUT_DIR}"/chunk_*.jsonl || true
echo "per-model aggregate:  python data_prep/score_rollouts.py ${OUT_DIR}/chunk_*.jsonl"
echo "vs another model:     ... --compare results/bench-<other>/chunk_*.jsonl"
