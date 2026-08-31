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
# bare "HF_TOKEN is unset" that is very hard to read.
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
        # Strip a trailing inline comment before the quotes, which judge.py's
        # loader does not do. The placeholder line written into .env for this
        # script carried a trailing "# ... Write perms" comment on the same line as the value; it
        # rode into HF_TOKEN and came back from the Hub client as "'ascii' codec
        # can't encode character '\uff0c'" -- the fullwidth comma, four layers
        # away from anything that looked like a .env problem.
        case "${v}" in
            \"*|\'*) ;;                        # quoted: a # inside is literal
            *[[:space:]]#*) v="$(trim "${v%%[[:space:]]#*}")" ;;
        esac
        v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
        if [ -n "${k}" ] && [ -n "${v}" ] && [ -z "${!k:-}" ]; then
            export "${k}=${v}"
        fi
    done < "${envf}"
}

load_dotenv "${ENV_FILE}"

# ============================ edit here ============================
REPO_ID="zoeyyy-wyd/agentic-tvg-grpo"              # must match hf_push.sh
DEST="./results/grpo-vanilla-1"                                    # download destination
# ===============================================================

die() { echo "error: $*" >&2; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is unset. Add HF_TOKEN=hf_xxx to ${ENV_FILE} (Read perms suffice), or export HF_TOKEN"
[[ "${HF_TOKEN}" =~ ^hf_[A-Za-z0-9]+$ ]] || die "HF_TOKEN contains non-token characters (length ${#HF_TOKEN}). The value in ${ENV_FILE} must be the bare token"
[[ "${REPO_ID}" == your-username/* ]] && die "REPO_ID is still the placeholder"
command -v hf >/dev/null || die "hf not found. First: pip install -U 'huggingface_hub[hf_xet]'"

export HF_TOKEN
unset HF_HUB_ENABLE_HF_TRANSFER || true
python3 -c "import hf_xet" 2>/dev/null \
    || echo "[warn] hf_xet not installed; downloads will be much slower: pip install -U 'huggingface_hub[hf_xet]'"

# `awk NR==1`, not `head -1`: head closes the pipe after one line, and when the
# writer is still going that SIGPIPE (141) becomes the pipeline's status under
# pipefail. awk reads to EOF. Called once and kept, too -- it was two network
# round trips for one answer.
whoami_line=$(hf auth whoami 2>&1 | awk 'NR==1')
echo "account: ${whoami_line}"
case "${whoami_line}" in
    *"Not logged in"*|*Error*|*Invalid*) die "token invalid or lacks read access to this private repo: ${whoami_line}" ;;
esac

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
    printf "repo : %.1f GiB\nfree disk: %.1f GiB\n", n/1048576, f/1048576
    if (f < n * 1.05) { print "error: not enough disk" > "/dev/stderr"; exit 1 }
}'

# --local-dir gives a plain tree instead of the blob+symlink cache layout, which
# is what verl and from_pretrained want to be pointed at. Interrupted downloads
# resume on re-run; hf verifies each file's hash, so a partial file is refetched
# rather than silently kept.
hf download "${REPO_ID}" --local-dir "${DEST}"

echo
echo "saved: ${DEST}  ($(du -sk "${DEST}" | awk '{printf "%.1f GiB", $1/1048576}'), $(find "${DEST}" -type f ! -path '*/.cache/*' | wc -l | tr -d ' ') files)"
if [ -f "${DEST}/latest_checkpointed_iteration.txt" ]; then
    echo "step: global_step_$(cat "${DEST}/latest_checkpointed_iteration.txt")"
    echo "resume: trainer.default_local_dir=${DEST}"
fi
