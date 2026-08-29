#!/usr/bin/env bash
# Download a checkpoint from a private HuggingFace repo. Run on your LOCAL machine.
#
# Pair of hf_push.sh. Put HF_TOKEN in .env, edit the two values below, then:
# bash hf_pull.sh
#
# Portability: this one runs on your laptop, so it avoids the GNU-only flags
# (du -sb, find -printf) that hf_push.sh uses freely on the Linux box, and it
# does not assume conda lives in /opt.

set -euo pipefail

# HF_TOKEN comes from the repo-root .env (gitignored, chmod 600), not from a
# literal here: this script is tracked, so an earlier revision's hardcoded
# token went to GitHub with it and stayed live in the history.
#
# Same file and same precedence as agentic_tvg/judge.py::_load_dotenv -- a
# value already exported in the shell wins. One deliberate difference: an
# exported-but-empty var falls through to .env here, where judge.py would
# defer to it, because deferring turns a set-but-blank HF_TOKEN into a
# "HF_TOKEN 没设" that is very hard to read.
ENV_FILE="$(dirname "$0")/.env"

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }

load_dotenv() {
    local envf="$1" line k v
    [ -f "${envf}" ] || return 0
    while IFS= read -r line || [ -n "${line}" ]; do
        line="$(trim "${line}")"
        case "${line}" in ''|'#'*) continue ;; esac
        [[ "${line}" == *=* ]] || continue
        line="${line#export }"
        k="$(trim "${line%%=*}")"
        v="$(trim "${line#*=}")"
        v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
        if [ -n "${k}" ] && [ -n "${v}" ] && [ -z "${!k:-}" ]; then
            export "${k}=${v}"
        fi
    done < "${envf}"
}

load_dotenv "${ENV_FILE}"

# ============================ 改这里 ============================
REPO_ID="zoeyyy-wyd/agentic-tvg-grpo"              # 跟 hf_push.sh 里填的一致
DEST="./results/grpo-vanilla-1"                                    # 下载到哪
# ===============================================================

die() { echo "错误: $*" >&2; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN 没设。在 ${ENV_FILE} 里加一行 HF_TOKEN=hf_xxx（Read 权限就够），或者 export HF_TOKEN"
[[ "${REPO_ID}" == your-username/* ]] && die "REPO_ID 还是占位符"
command -v hf >/dev/null || die "找不到 hf。先跑: pip install -U 'huggingface_hub[hf_xet]'"

export HF_TOKEN
unset HF_HUB_ENABLE_HF_TRANSFER || true
python3 -c "import hf_xet" 2>/dev/null \
    || echo "[warn] 未装 hf_xet，下载会明显变慢: pip install -U 'huggingface_hub[hf_xet]'"

echo "帐号: $(hf auth whoami 2>&1 | head -1)"
[[ "$(hf auth whoami 2>&1 | head -1)" == *"Not logged in"* ]] && die "token 无效或没有这个私有仓库的读权限"

# Ask the Hub how big this is before committing the disk to it -- running out
# of space 15G into a 17G download leaves a truncated tree that looks complete.
need_kb=$(python3 - "${REPO_ID}" <<'PY'
import sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1], files_metadata=True)
print(sum(f.size or 0 for f in info.siblings) // 1024)
PY
)
mkdir -p "${DEST}"
free_kb=$(df -Pk "${DEST}" | tail -1 | awk '{print $4}')
awk -v n="${need_kb}" -v f="${free_kb}" 'BEGIN{
    printf "仓库 : %.1f GiB\n剩余磁盘: %.1f GiB\n", n/1048576, f/1048576
    if (f < n * 1.05) { print "错误: 磁盘不够" > "/dev/stderr"; exit 1 }
}'

# --local-dir gives a plain tree instead of the blob+symlink cache layout, which
# is what verl and from_pretrained want to be pointed at. Interrupted downloads
# resume on re-run; hf verifies each file's hash, so a partial file is refetched
# rather than silently kept.
hf download "${REPO_ID}" --local-dir "${DEST}"

echo
echo "落盘: ${DEST}  ($(du -sk "${DEST}" | awk '{printf "%.1f GiB", $1/1048576}'), $(find "${DEST}" -type f ! -path '*/.cache/*' | wc -l | tr -d ' ') 个文件)"
if [ -f "${DEST}/latest_checkpointed_iteration.txt" ]; then
    echo "步数: global_step_$(cat "${DEST}/latest_checkpointed_iteration.txt")"
    echo "续训: trainer.default_local_dir=${DEST}"
fi
