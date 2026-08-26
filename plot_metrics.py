#!/usr/bin/env python
"""Plot training curves from run_sft.sh console logs.

Parses the trainer's per-step lines ("step:N - train/loss:... - ...") and
writes <log>.png next to each log. Metrics are also written as tensorboard
events (logs/tb/<EXP_NAME>, see run_sft.sh) -- this script is the
no-server-needed complement for quick PNGs.

Usage:
    python plot_metrics.py                     # newest logs/*.log with step lines
    python plot_metrics.py logs/sft_mix_*.log  # explicit log(s)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAIR = re.compile(r"([\w/().\-]+):(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\b")


def parse(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    series: dict[str, tuple[list[float], list[float]]] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("step:"):
            continue
        pairs = {}
        for k, v in PAIR.findall(line):
            try:
                pairs[k] = float(v)
            except ValueError:
                pass
        step = pairs.pop("step", None)
        if step is None:
            continue
        for k, v in pairs.items():
            series.setdefault(k, ([], []))
            series[k][0].append(step)
            series[k][1].append(v)
    return series


def plot(path: Path) -> Path | None:
    series = parse(path)
    if not series:
        return None
    panels = [
        ("loss", [k for k in series if k.endswith("/loss")]),
        ("memory (GB)", [k for k in series if "memory" in k]),
        ("lr", [k for k in series if "lr" in k.lower()]),
        ("other", [k for k in series
                   if not k.endswith("/loss") and "memory" not in k and "lr" not in k.lower()]),
    ]
    panels = [(t, ks) for t, ks in panels if ks]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, keys) in zip(axes, panels):
        for k in sorted(keys):
            steps, vals = series[k]
            style = "o-" if len(steps) < 20 else "-"
            ax.plot(steps, vals, style, label=k, markersize=3, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(path.name, fontsize=11)
    fig.tight_layout()
    out = path.with_suffix(path.suffix + ".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    if len(sys.argv) > 1:
        logs = [Path(a) for a in sys.argv[1:]]
    else:
        candidates = sorted(Path("logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        logs = next(([p] for p in candidates if parse(p)), [])
        if not logs:
            sys.exit("no log with step lines found under logs/")
    for log in logs:
        out = plot(log)
        print(f"{log} -> {out if out else 'no step lines, skipped'}")


if __name__ == "__main__":
    main()
