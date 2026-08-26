#!/usr/bin/env bash
# Fail-fast guards for the two things that silently ruin a long run.
# Sourced (not executed) by run_sft.sh / run_grpo.sh.
#
# 1. Wrong conda env -> ModuleNotFoundError 20 minutes in. RUNBOOK's pitfall #1.
# 2. Missing LD_PRELOAD -> torchcodec's dlopen fails on the first video decode,
#    not at import (ENVIRONMENT.md §7). `conda env config vars set` puts it on
#    the env, so it is present in any shell activated after that was run -- but
#    a shell activated *before* it, or one that inherited a stripped environment,
#    still has it empty. That case is recoverable, so we repair it and warn
#    rather than abort.

_pf_die() { echo -e "\033[1;31m[preflight] $*\033[0m" >&2; exit 1; }
_pf_warn() { echo -e "\033[1;33m[preflight] $*\033[0m" >&2; }

[ -n "${CONDA_PREFIX:-}" ] || _pf_die "no conda env active. Run: conda activate verl"

if [ "${CONDA_DEFAULT_ENV:-}" != "verl" ]; then
    _pf_die "conda env is '${CONDA_DEFAULT_ENV:-none}', expected 'verl'. Run: conda activate verl"
fi

_pf_libstdcxx="${CONDA_PREFIX}/lib/libstdc++.so.6"
case "${LD_PRELOAD:-}" in
    *libstdc++*) ;;
    *)
        [ -e "$_pf_libstdcxx" ] || _pf_die "LD_PRELOAD unset and $_pf_libstdcxx is missing. Re-run env_setup/setup_verl_env.sh"
        export LD_PRELOAD="${_pf_libstdcxx}${LD_PRELOAD:+:$LD_PRELOAD}"
        _pf_warn "LD_PRELOAD was unset -- repaired for this run (ENVIRONMENT.md §7)."
        _pf_warn "This shell predates the env var; 'conda deactivate && conda activate verl' makes it permanent."
        ;;
esac

# Cheap end-to-end proof that video decoding works, ~1 s. Catches the §7 failure
# here instead of inside the trainer's dataloader.
python - <<'PY' || _pf_die "torchcodec cannot decode. See ENVIRONMENT.md §7."
import os, sys, tempfile
import av, numpy as np
from torchcodec.decoders import VideoDecoder
from qwen_vl_utils.vision_process import get_video_reader_backend

backend = get_video_reader_backend()
if backend != "torchcodec":
    sys.exit(f"qwen-vl-utils picked {backend!r}, not torchcodec")
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "p.mp4")
    with av.open(p, mode="w") as out:
        st = out.add_stream("libx264", rate=10)
        st.width, st.height, st.pix_fmt = 64, 64, "yuv420p"
        for i in range(10):
            out.mux(st.encode(av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), i * 25, dtype=np.uint8), format="rgb24")))
        out.mux(st.encode(None))
    VideoDecoder(p)[0]
PY

echo "[preflight] env=verl | LD_PRELOAD ok | torchcodec decode ok"
