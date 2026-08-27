#!/usr/bin/env python
"""Sample memory every 20s, splitting the three hypotheses apart.

RSS double-counts pages shared between a parent and its forked children, so a
sum over Ray workers can grow while nothing is actually leaking. PSS divides
each shared page by its number of sharers, so summing PSS is honest -- that is
the number to trust here.

Columns: ts, total_used_gb, shm_gb, pss_total_gb, pss_actors_gb,
         pss_forked_gb, n_proc, top3
"""
import os, time, sys

def pss_gb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1048576
    except (OSError, ValueError):
        pass
    return 0.0

def cmd(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""

def ppid_of(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0

def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.strip().split()[0]) / 1048576
    used = d["MemTotal"] - d["MemAvailable"]
    return used, d.get("Shmem", 0.0)

print("ts total_used_gb shm_gb pss_total_gb pss_actors_gb pss_forked_gb n_proc top3", flush=True)
while True:
    procs = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        c = cmd(pid)
        if ("ray::" in c or "verl.trainer" in c or "VLLM" in c) and "memsample" not in c:
            procs[int(pid)] = (pss_gb(pid), ppid_of(pid), c[:40])
    # A worker whose parent is also in the set was forked by it (DataLoader etc.);
    # everything else was launched by raylet and is a real actor.
    actors = {p: v for p, v in procs.items() if v[1] not in procs}
    forked = {p: v for p, v in procs.items() if v[1] in procs}
    used, shm = meminfo()
    tot = sum(v[0] for v in procs.values())
    top = sorted(procs.items(), key=lambda kv: -kv[1][0])[:3]
    top_s = "|".join(f"{p}:{v[0]:.1f}G:{v[2].split()[0][-18:]}" for p, v in top)
    print(f"{time.strftime('%H:%M:%S')} {used:.1f} {shm:.1f} {tot:.1f} "
          f"{sum(v[0] for v in actors.values()):.1f} "
          f"{sum(v[0] for v in forked.values()):.1f} {len(procs)} {top_s}", flush=True)
    time.sleep(20)
