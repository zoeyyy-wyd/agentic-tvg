#!/usr/bin/env python
"""Plot training curves from run_sft.sh console logs.

GRPO logs are a different shape -- 106 metrics per step instead of 10 -- and
have their own script, plot_grpo.py, which reuses parse()/write_csv() here.

Parses the trainer's per-step lines ("step:N - train/loss:... - ...") and
writes <log>.png next to each log. Metrics are also written as tensorboard
events (logs/tb/<EXP_NAME>, see run_sft.sh) -- this script is the
no-server-needed complement for quick PNGs.

Usage:
    python plot_sft.py                     # newest logs/*.log with step lines
    python plot_sft.py logs/sft_mix_*.log  # explicit log(s), one PNG each
    python plot_sft.py LOG -o results/sft-mix/curves.png   # explicit destination
    python plot_sft.py LOG --csv results/sft-mix/metrics.csv  # numbers, not pixels

A resumed run leaves one log per attempt. Concatenate them oldest-first for a
single whole-run curve -- repeated steps keep their LAST value, which is the
one that survived the rollback, so no hand-trimming is needed:

    cat logs/sft_mix_20260826_101821.log logs/sft_mix_20260826_175728.log \
        > logs/sft_mix_full.log
    python plot_sft.py logs/sft_mix_full.log

Token and memory metrics are written to `--csv` but get no panel -- the figure
is loss / lr / grad_norm / mfu. See NO_PANEL below for why.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# verl prints some metrics bare and others through their numpy repr:
#   actor/entropy:1.078
#   actor/pg_loss:np.float64(0.00832)
# Requiring a digit right after the colon silently drops the wrapped ones -- on
# a GRPO log that is 32 of 106 metrics, and they are the interesting ones
# (pg_loss, kl_loss, grad_norm, lr). The optional np.<type>( prefix fixes it.
# The @ matters: val tags end in "/mean@1", and without @ in the class the
# key match stops at the "1" before the colon -- every val series silently
# collapses into a garbage series named "1".
PAIR = re.compile(
    r"([\w/().\-@]+):(?:np\.\w+\()?(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
# tqdm writes its bar to the same tee'd stream, so the metrics line usually
# arrives glued to a progress bar: "Epoch 1/2: 53%|...|32/60 [08:31<...]step:33
# - train/loss:...". Anchor on step:N and slice, never parse the prefix -- the
# bar's own "[08:31<31:52" would otherwise register as series named 08 and 31.
STEP = re.compile(r"\bstep:\d+")
# Bookkeeping, not curves -- kept in --csv, denied a panel. Token counters:
# global_tokens is flat by construction (dynamic-bsz packs every step to the
# same token budget) and total_tokens(B) is a per-process cumulative that
# resets to 0 on resume, so its only visible feature is a sawtooth artifact.
# Memory: allocated/reserved are flat once the first steps settle, and
# cpu_memory_used mostly tracks page cache (checkpoint writes show up as
# cliffs that mean nothing about the model). Read them from the CSV when
# chasing an OOM; they crowd out the four curves that carry the run.
NO_PANEL = re.compile(r"token|memory", re.I)


def parse(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    # step -> metric -> value. A resumed run replays the steps that were rolled
    # back with the checkpoint (here 26-49, discarded when we resumed from 25),
    # so the LAST occurrence of a step is the one that survived training: dict
    # assignment in file order gives exactly that, and makes concatenating a
    # crashed log with its resume log the correct way to plot the whole run.
    by_step: dict[float, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP.search(line)
        if not m:
            continue
        pairs = {}
        for k, v in PAIR.findall(line[m.start():]):
            try:
                pairs[k] = float(v)
            except ValueError:
                pass
        step = pairs.pop("step", None)
        if step is None:
            continue
        by_step.setdefault(step, {}).update(pairs)

    series: dict[str, tuple[list[float], list[float]]] = {}
    for step in sorted(by_step):
        for k, v in by_step[step].items():
            series.setdefault(k, ([], []))
            series[k][0].append(step)
            series[k][1].append(v)
    return series


def plot(path: Path, out: Path | None = None) -> Path | None:
    series = parse(path)
    if not series:
        return None
    plotted = [k for k in series if not NO_PANEL.search(k)]
    grouped = [
        ("loss", [k for k in plotted if k.endswith("/loss")]),
        # Empty while NO_PANEL filters memory out; kept so the grouping comes
        # back correctly if that filter is ever relaxed.
        ("memory (GB)", [k for k in plotted if "memory" in k]),
        ("lr", [k for k in plotted if "lr" in k.lower()]),
    ]
    claimed = {k for _, ks in grouped for k in ks}
    # Everything else gets its own panel rather than a shared "other": mfu
    # (0.32) and grad_norm (0.2) are invisible next to global_tokens (~230K),
    # and those two are the ones worth reading.
    panels = [(t, ks) for t, ks in grouped if ks]
    panels += [(k, [k]) for k in sorted(plotted) if k not in claimed]
    heights = [3 if len(ks) > 1 else 2 for _, ks in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, sum(heights)), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, keys) in zip(axes, panels):
        for k in sorted(keys):
            steps, vals = series[k]
            # A sparse series (val/loss: 5 points against train/loss's 120) is
            # the one you came to read, and thin 3pt markers bury it under the
            # noisy dense line it sits on top of. Draw it heavier and above.
            sparse = len(steps) < 20
            ax.plot(steps, vals, "o-" if sparse else "-", label=k,
                    markersize=7 if sparse else 3,
                    linewidth=2.2 if sparse else 1.2,
                    zorder=3 if sparse else 2)
            if sparse:
                for x, y in zip(steps, vals):
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(path.name, fontsize=11)
    fig.tight_layout()
    out = out or path.with_suffix(path.suffix + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def write_csv(series: dict, out: Path) -> Path:
    """One row per step, one column per metric -- the form that outlives the log."""
    steps = sorted({int(x) for k in series for x in series[k][0]})
    cols = sorted(series)
    lut = {k: dict(zip(map(int, series[k][0]), series[k][1])) for k in series}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + cols)
        for st in steps:
            w.writerow([st] + [f"{lut[c][st]:.6g}" if st in lut[c] else "" for c in cols])
    return out


def main() -> None:
    argv = sys.argv[1:]
    out = None
    csv_out = None
    for flag, target in (("-o", "png"), ("--out", "png"), ("--csv", "csv")):
        if flag in argv:
            i = argv.index(flag)
            value = Path(argv[i + 1])
            del argv[i : i + 2]
            if target == "png":
                out = value
            else:
                csv_out = value
    if len(argv) > 1 and (out is not None or csv_out is not None):
        sys.exit("-o/--csv take a single log; got several")
    if argv:
        logs = [Path(a) for a in argv]
    else:
        candidates = sorted(Path("logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        logs = next(([p] for p in candidates if parse(p)), [])
        if not logs:
            sys.exit("no log with step lines found under logs/")
    for log in logs:
        written = plot(log, out)
        print(f"{log} -> {written if written else 'no step lines, skipped'}")
        if csv_out is not None and written:
            print(f"{log} -> {write_csv(parse(log), csv_out)}")


if __name__ == "__main__":
    main()
