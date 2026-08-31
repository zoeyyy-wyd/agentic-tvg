#!/usr/bin/env bash
# Upload a checkpoint directory to a private HuggingFace repo. Run on the server.
#
# Why the Hub rather than scp: this box has a 40 MB/s uplink and 3ms to HF, so
# the direct link is not what is slow -- scp is one TCP stream, and on a
# high-latency path one stream stalls on its window long before it saturates
# the line. The Hub's client opens many, and Xet chunks content so a re-run
# after a failure re-sends only what never landed.
#
# Put HF_TOKEN in .env, edit the two values below, then: bash hf_push.sh
# (add -y to skip the prompt)

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
REPO_ID="zoeyyy-wyd/agentic-tvg-grpo-final"                   # target repo; auto-created private if missing
LOCAL_PATH="./results/grpo-vanilla"                # directory or file to upload
# ===============================================================

# The whole ckpt/ dir, not just global_step_N: verl reads
# latest_checkpointed_iteration.txt from the parent when it resumes, and a
# checkpoint you cannot resume from is not a backup.

COMMIT_MSG="${COMMIT_MSG:-$(basename "${LOCAL_PATH}") $(date -u +%Y-%m-%dT%H:%MZ)}"
export PATH="/opt/miniconda3/bin:${PATH}"
cd "$(dirname "$0")"

die() { echo "error: $*" >&2; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is unset. Add HF_TOKEN=hf_xxx to ${ENV_FILE} (Write token: https://huggingface.co/settings/tokens), chmod 600 ${ENV_FILE}; or export HF_TOKEN"
[[ "${HF_TOKEN}" =~ ^hf_[A-Za-z0-9]+$ ]] || die "HF_TOKEN contains non-token characters (length ${#HF_TOKEN}). The value in ${ENV_FILE} must be the bare token"
[[ "${REPO_ID}" == your-username/* ]] && die "REPO_ID is still the placeholder -- set your own username"
[ -e "${LOCAL_PATH}" ] || die "LOCAL_PATH does not exist: ${LOCAL_PATH}"
command -v hf >/dev/null || die "hf not found. pip install -U 'huggingface_hub[hf_xet]'"

# Exported, not passed as --token: the flag would put the secret in the process
# table where `ps` shows it to every user on the box.
export HF_TOKEN

# hf_xet is the current upload backend and is already installed here. The older
# HF_HUB_ENABLE_HF_TRANSFER path is not an addition to it -- setting it routes
# around Xet and loses the chunk-level resume this script relies on.
unset HF_HUB_ENABLE_HF_TRANSFER || true
python -c "import hf_xet" 2>/dev/null \
    || echo "[warn] hf_xet not installed; uploads take the slow path: pip install -U 'huggingface_hub[hf_xet]'"

# `awk NR==1`, not `head -1`: head closes the pipe after one line, and when the
# writer is still going that SIGPIPE (141) becomes the pipeline's status under
# pipefail. awk reads to EOF. Called once and kept, too -- it was two network
# round trips for one answer.
whoami_line=$(hf auth whoami 2>&1 | awk 'NR==1')
echo "account: ${whoami_line}"
case "${whoami_line}" in
    *"Not logged in"*|*Error*|*Invalid*) die "token invalid or expired: ${whoami_line}" ;;
esac

bytes=$(du -sb "${LOCAL_PATH}" | cut -f1)
gib=$(awk "BEGIN{printf \"%.1f\", ${bytes}/2^30}")
# 40 MB/s measured against the Hub from this host on 2026-08-28.
eta=$(awk "BEGIN{printf \"%.0f\", ${bytes}/40/1048576/60}")
# One awk pass instead of `sort -rn | head -1`. That pipeline was the bug: sort
# writes in chunks, head -1 exits after the first, and the resulting SIGPIPE ->
# pipefail -> set -e killed the script from inside a $() with nothing printed --
# right after the account line, on 49 of 200 runs measured here. Same output shape
# (size TAB path), and it does not sort 268 lines to keep one.
biggest=$(find "${LOCAL_PATH}" -type f -printf "%s\t%p\n" \
    | awk -F'\t' '$1 > m {m = $1; p = $2} END {print m "\t" p}')
echo "upload: ${LOCAL_PATH}  ->  ${REPO_ID}  (private)"
echo "size  : ${gib} GiB, $(find "${LOCAL_PATH}" -type f | wc -l) files, est. ~${eta} min"
echo "largest file: $(awk -v b="${biggest%%	*}" 'BEGIN{printf "%.1f GiB", b/2^30}')  ${biggest##*	}"
awk -v b="${biggest%%	*}" 'BEGIN{ if (b > 50*2^30) { print "error: single file exceeds the Hub 50GiB limit" > "/dev/stderr"; exit 1 } else if (b > 20*2^30) print "[warn] single file over 20GiB; the Hub recommends splitting" }'

if [ "${1:-}" != "-y" ]; then
    read -rp "Upload this to HuggingFace? [y/N] " ok
    [[ "${ok}" == [yY] ]] || { echo "cancelled"; exit 0; }
fi

# Re-running after a failure is the recovery path: Xet skips chunks the server
# already has, so a resumed upload costs only what never arrived.
hf upload "${REPO_ID}" "${LOCAL_PATH}" . \
    --repo-type=model --private --commit-message="${COMMIT_MSG}"

echo
echo "done: https://huggingface.co/${REPO_ID}"
echo "pull locally: set REPO_ID in hf_pull.sh to ${REPO_ID}"
