#!/usr/bin/env bash
# Who is holding /dev/shm, and is it growing?
#
# Four CPU OOMs during GRPO (2026-08-27/28) traced the growth to /dev/shm:
# in a controlled 8-step run the node's used memory rose 5.7 G while shm rose
# 6.1 G -- essentially all of it. What was never established is *which*
# process holds those segments, because by the time we looked the run was
# dead. Run this against a live run to settle it.
#
#   bash diagnose_shm.sh          # one snapshot
#   bash diagnose_shm.sh 300      # snapshot now, again in 300s, report the delta
#
# Reading the result:
#   ray::TaskRunner / ray::WorkerDict dominate -> Ray's object store. Cap it
#     with RAY_object_store_memory / `ray start --object-store-memory`; Ray
#     spills to disk instead of eating RAM.
#   ray::AgentLoopWorker dominates            -> decoded video tensors are not
#     being released. Turn down rollout.agent.num_workers (default 8).
#   VLLM::Worker dominates                    -> the engine's own IPC buffers;
#     look at gpu_memory_utilization and the sleep/wake cycle instead.
set -uo pipefail

snapshot() {
    echo "--- $(date +%H:%M:%S) ---"
    # Three different numbers, routinely conflated:
    #   df /dev/shm  - the tmpfs, INCLUDING segments unlinked but still open
    #   du /dev/shm  - only what is still named; misses every "(deleted)" one
    #   Shmem        - df's number PLUS shared anonymous maps (CUDA IPC etc),
    #                  so a gap between the two is NOT /dev/shm growth
    df -h /dev/shm | tail -1 | awk '{printf "  /dev/shm(df)  %6s of %s\n", $3, $2}'
    awk '/^Shmem:/{printf "  Shmem         %6.1f G   <- includes non-tmpfs shared maps\n",$2/1048576}' /proc/meminfo
    free -g | awk 'NR==2{printf "  node          %s G used of %s G\n", $3, $2}'
    echo "  segments held, by process:"
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
        # A pid can vanish between listing /proc and reading it, and the
        # failing redirect is reported by the shell itself -- 2>/dev/null on tr
        # does not catch it, so guard with -r and silence the block.
        [ -r "/proc/$p/cmdline" ] || continue
        c=$({ tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null) || continue
        case "$c" in *ray::*|*VLLM*|*verl.trainer*) ;; *) continue;; esac
        case "$c" in *diagnose_shm*) continue;; esac
        n=$(ls -l "/proc/$p/fd" 2>/dev/null | grep -c '/dev/shm/') || n=0
        [ "$n" -gt 0 ] && printf "    %5d segs  pid=%-8s %s\n" "$n" "$p" "$(echo "$c" | cut -c1-46)"
    done | sort -rn | head -12
    # Total distinct segments and their bytes -- the du is what actually counts
    # against RAM; the fd count above only says who is keeping them alive.
    echo "  named files: $(ls /dev/shm 2>/dev/null | wc -l), $(du -sh /dev/shm 2>/dev/null | cut -f1) (unlinked-but-open ones are invisible here -- trust df)"
}

snapshot
if [ "${1:-0}" -gt 0 ]; then
    echo
    echo "sleeping ${1}s ..."
    sleep "$1"
    echo
    snapshot
fi
