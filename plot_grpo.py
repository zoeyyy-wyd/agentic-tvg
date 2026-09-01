#!/usr/bin/env python
"""Plot GRPO training curves from run_grpo.sh console logs.

Separate from plot_sft.py because the two logs are different shapes, not
different data: SFT emits ~10 metrics per step and plot_sft.py can afford to
blacklist a few and panel the rest. GRPO emits **106**, so this one works from
a whitelist -- the dozen or so that answer "is it learning?", "is it healthy?"
and "is it gaming the reward?" -- and leaves the other ninety in the CSV.

Parsing is shared (parse/write_csv imported from plot_sft): both trainers print
the same `step:N - k:v - k:v` lines, glued to a tqdm bar.

Two things this has to get right that the SFT plotter did not:

* **Mixed sample rates.** Training metrics land every step, `val-*` only every
  test_freq steps. A 5-point val series drawn thin on top of a 267-point train
  series is invisible, so sparse series are drawn heavy and value-labelled.
* **Mixed scales.** response_length is in thousands, num_turns is ~4,
  kl_loss is ~1e-3. Only metrics that genuinely share a scale share a panel.

Usage:
    python plot_grpo.py                          # newest results/grpo*/console_*.log
    python plot_grpo.py LOG -o curves.png --csv metrics.csv
    python plot_grpo.py results/grpo-v2/tb -o curves.png        # TB event DIR

The TB-directory mode exists because console logs are mortal (one was rm'd
mid-run on 2026-08-28) while the tensorboard events under results/<run>/tb/
survive every crash and resume. All event files in the dir are merged in
mtime order with last-write-wins per step -- the same convention as
concatenating console logs, so a resumed run's replayed steps take precedence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_sft import parse, write_csv  # noqa: E402

# verl namespaces the val metrics by data source, e.g.
# `val-core/longvt_longvt-val/acc/mean@1`. Nobody needs to read that on a legend.
VAL_RE = re.compile(r"^val-(?:core|aux)/[^/]+/(.+?)/mean@1$")


def short(key: str) -> str:
    m = VAL_RE.match(key)
    return f"val/{m.group(1)}" if m else key


def val_key(series: dict, name: str) -> str | None:
    """Find `val-core|val-aux/<source>/<name>/mean@1` without hardcoding source."""
    for k in series:
        m = VAL_RE.match(k)
        if m and m.group(1) == name:
            return k
    return None


# (title, [metric keys]) -- keys grouped ONLY where the scale is genuinely shared.
def panels_for(series: dict) -> list[tuple[str, list[str]]]:
    v = lambda n: val_key(series, n)  # noqa: E731
    groups = [
        # Is it learning? Train reward every step, val reward every test_freq --
        # same 0..2 scale, so they belong together, and the gap between them is
        # the whole point: train up while val is flat means reward hacking.
        ("reward (train vs val)", ["critic/rewards/mean", v("reward")]),
        # What the reward is made of. All 0..1, all sparse.
        ("val quality", [v("acc"), v("format_score"), v("evidence_iou")]),
        ("actor/pg_loss", ["actor/pg_loss"]),
        ("actor/kl_loss", ["actor/kl_loss"]),
        # Collapse shows up here first: entropy falling off a cliff means the
        # policy stopped exploring, usually just before reward flatlines.
        ("actor/entropy", ["actor/entropy"]),
        ("actor/grad_norm", ["actor/grad_norm"]),
        ("actor/lr", ["actor/lr"]),
        # NOTE (2026-09-01): this panel CANNOT show signal loss. GRPO's
        # advantage is group-normalized, so its batch mean is ~0 by
        # construction (measured: -0.01..-0.03 all run) whatever happens to
        # the pool. The quantity that does show saturation -- within-group
        # spread / mastered share -- is not in the TB stream at all; get it
        # mid-run with data_prep/analyze_groups.py --signal on rollouts/.
        ("critic/advantages/mean", ["critic/advantages/mean"]),
        # Length inflation is the classic way to game a format+accuracy reward.
        ("response_length (tokens)", ["response_length/mean", "response_length/max"]),
        ("num_turns/mean", ["num_turns/mean"]),
        ("ratios", ["response/aborted_ratio", "response_length/clip_ratio"]),
        ("perf/time_per_step (s)", ["perf/time_per_step"]),
        # The failure that actually stopped this run once: CPU, not GPU.
        ("cpu_memory_used_gb", ["actor/perf/cpu_memory_used_gb"]),
    ]
    out = []
    for title, keys in groups:
        present = [k for k in keys if k and k in series]
        if present:
            out.append((title, present))
    return out


def tb_series(d: Path) -> dict:
    """Merge every events file under d into parse()-shaped series."""
    from tensorboard.backend.event_processing import event_accumulator
    by_step = {}
    for f in sorted(d.glob("events*"), key=lambda q: q.stat().st_mtime):
        ea = event_accumulator.EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for ev in ea.Scalars(tag):
                by_step.setdefault(ev.step, {})[tag] = ev.value
    series = {}
    for step in sorted(by_step):
        for k, v in by_step[step].items():
            series.setdefault(k, ([], []))
            series[k][0].append(step)
            series[k][1].append(v)
    return series


def load(path: Path) -> dict:
    return tb_series(path) if path.is_dir() else parse(path)


def plot(path: Path, out: Path | None = None) -> Path | None:
    series = load(path)
    panels = panels_for(series)
    if not panels:
        return None

    heights = [3 if len(ks) > 1 else 2 for _, ks in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, sum(heights)), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, keys) in zip(axes, panels):
        for k in keys:
            steps, vals = series[k]
            sparse = len(steps) < 20
            ax.plot(steps, vals, "o-" if sparse else "-", label=short(k),
                    markersize=7 if sparse else 2.5,
                    linewidth=2.2 if sparse else 1.2,
                    zorder=3 if sparse else 2)
            if sparse and len(keys) <= 2:
                for x, y in zip(steps, vals):
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{path.name}  ({len(series)} metrics logged, {len(panels)} shown)",
                 fontsize=11)
    fig.tight_layout()
    out = out or (path / "curves.png" if path.is_dir() else path.with_suffix(path.suffix + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    argv = sys.argv[1:]
    out = csv_out = None
    for flag, target in (("-o", "png"), ("--out", "png"), ("--csv", "csv")):
        if flag in argv:
            i = argv.index(flag)
            value = Path(argv[i + 1])
            del argv[i : i + 2]
            if target == "png":
                out = value
            else:
                csv_out = value
    if argv:
        logs = [Path(a) for a in argv]
    else:
        cands = sorted(Path("results").glob("grpo*/console_[0-9]*.log"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        logs = next(([p] for p in cands if parse(p)), [])
        if not logs:
            sys.exit("no results/grpo*/console_*.log with step lines found")
    for log in logs:
        written = plot(log, out)
        print(f"{log} -> {written if written else 'no step lines, skipped'}")
        if csv_out is not None and written:
            # Every metric, not just the plotted ones -- the whitelist decides
            # what is worth a panel, never what is worth keeping.
            print(f"{log} -> {write_csv(load(log), csv_out)}")


if __name__ == "__main__":
    main()
