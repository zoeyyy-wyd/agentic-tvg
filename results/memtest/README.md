# CPU OOM diagnosis, 2026-08-27

Why GRPO died three times in a row, and what fixed it. Full write-up in
`GRPO_NOTES.md` §6; this directory holds the measurements behind it.

| file | what |
|---|---|
| `memsample.py` | the sampler: total/shm/PSS every 20 s, split into raylet-launched actors vs their forked children |
| `mem_pss.log` | run A — 8 steps, `param_offload=True`, glibc vars set |
| `mem_offload.log` | run B — 4 steps, `param_offload=False`, same glibc vars |
| `console.log` | run A's trainer output |

Both runs used batch 8 x K=16 = 128 trajectories, no validation, no
checkpointing, so the only variable between them is offload.

## Result

The fix was two glibc environment variables (`MALLOC_MMAP_THRESHOLD_`,
`MALLOC_TRIM_THRESHOLD_`), now set in `run_grpo.sh`. Run A survived 8 steps
where the same config had died at step 4; memory stopped climbing and settled
at 99-101 GB.

Run B tested the hypothesis that `param_offload=True` was holding ~45 GB of
host RAM. It was not: peak 171.3 vs 175.3 GB, WorkerDict 37.9 vs 38.4 GB.
Offload was kept off anyway, for speed — 785 s/step vs 876 s.

## Use PSS, not RSS

`ps` charges a shared page to every process that maps it, so summing RSS over
a parent and its forked children multiplies the shared part. PSS divides each
page by its number of sharers; summing PSS gives real physical usage. The
"TaskRunner is using 75 GB" reading that sent this investigation down a wrong
path was an RSS artifact.
