#!/usr/bin/env python
"""Build the RFT (stage 3) set from GRPO rollout dumps. README "RFT recipe".

Raw material: results/<grpo run>/rollouts/<step>.jsonl (round 1:
results/grpo-vanilla/rollouts_grpo267/) -- one row per
sampled trajectory with the flattened prompt (`input`), the flattened
generation incl. tool responses (`output`), and the reward breakdown. The
recipe, per README:

  1. keep score > --min-score (default 1.5; in practice acc=1 traces);
  2. per question, up to --traces-per-q answer-distinct traces
     (render_traces.pick_traces), preferring late steps -- traces are sorted
     by (step desc, score desc) before picking, so K=16 x 2 epochs of easy
     prompts cannot flood the set;
  3. rebuild <image> placeholders + crop frames by re-executing the logged
     crop_video call through the SAME code path as the RL tool
     (_normalize_window -> sample_frames -> crop_response_text) and requiring
     the rebuilt response text to be byte-identical to the logged one. Any
     drift (decode nondeterminism, changed constants, wrong duration) fails
     that equality and drops the trace -- zero train/serve skew, checked
     rather than assumed.

Faithfulness choices, deliberately different from render_traces.py: these
traces come from OUR OWN rollouts, so the first assistant turn is kept
verbatim (the model's own <think> + tool_call bytes, not a canonical
re-render) and the tool message carries the logged response text. Only
<image>/<video> literals are scrubbed from model-authored text (the rft_9397
lesson, DATA.md §7.2). Prompts are rebuilt from agentic_tvg.prompts with the
duration from rl_train.parquet -- the same value the RL tool was created with
-- and the row's `input` is required to contain that rendered duration.

Traceback: question text -> rl_train.parquet is unique (README, verified
here); it supplies video_path, duration, gt and evidence segment.

Outputs (all prefixed by --prefix, default "rft_v2"; round 1 used "rft"):
<prefix>_train.parquet / <prefix>_val.parquet (run_sft.sh schema: messages/
images/videos/tools/extra_info), frames under --out/frames_<prefix>/,
<prefix>_selection.json (stats + per-question picks), and <prefix>_review.md -- a
readable sample of picked score > 1.8 traces for the mandatory hand-read
before training (RFT bakes reward quirks into weights harder than RL).

Usage:
    python data_prep/extract_rft.py --plan-only     # selection stats only
    python data_prep/extract_rft.py                 # full render (~1-2 s/trace)
    TRAIN_FILES=data/processed/rft_v2_train.parquet VAL_FILES=data/processed/rft_v2_val.parquet \
        MODEL_PATH=results/grpo-v2/merged EXP_NAME=rft_v2 bash run_sft.sh
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_traces import ANSWER_RE, TOOL_CALL_RE, pick_traces, strip_mm_tags  # noqa: E402

from agentic_tvg.constants import (  # noqa: E402
    CROP_MAX_PIXELS,
    CROP_MIN_PIXELS,
    CROP_NUM_FRAMES,
    GLOBAL_MAX_PIXELS,
    GLOBAL_MIN_PIXELS,
    GLOBAL_NUM_FRAMES,
    MIN_CROP_SECONDS,
)
from agentic_tvg.crop_video_tool import (  # noqa: E402
    _normalize_window,
    build_crop_video_schema,
    crop_response_text,
)
from agentic_tvg.prompts import build_system_prompt, build_user_prompt  # noqa: E402
from agentic_tvg.video_frames import sample_frames  # noqa: E402

# The question sits between fixed byte sequences of the rendered user prompt;
# greedy (.*) is safe because each anchors once per prompt, and the question
# itself may contain quotes.
QUESTION_RE = re.compile(r'Question: "(.*)"\nAnswer the question based on the video\.', re.DOTALL)
# What the chat template flattens between the tool call and the final turn.
AFTER_TOOL_RE = re.compile(r"\nuser\n<tool_response>\n(.*?)\n</tool_response>\nassistant\n(.*)\Z", re.DOTALL)


def parse_rollout(d: dict) -> tuple[dict | None, str]:
    """One dump row -> {question, a1, window, resp, final, answer} or (None, why)."""
    out = d["output"]
    tcs = list(TOOL_CALL_RE.finditer(out))
    if len(tcs) != 1:
        return None, f"tool_calls_{len(tcs)}"
    tc = tcs[0]
    think1 = out[: tc.start()].rstrip()
    if not (think1.startswith("<think>") and think1.endswith("</think>")):
        return None, "bad_think1"
    try:
        call = json.loads(tc.group(1))
        args = call["arguments"]
        window = (float(args["start_time"]), float(args["end_time"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, "bad_tool_json"
    if call.get("name") != "crop_video":
        return None, "wrong_tool"
    m = AFTER_TOOL_RE.match(out[tc.end():])
    if not m:
        return None, "bad_structure"
    resp, final = m.group(1), m.group(2).strip()
    ans = ANSWER_RE.findall(final)
    if len(ans) != 1 or not final.startswith("<think>") or not final.endswith("</answer>"):
        return None, "bad_final"
    q = QUESTION_RE.search(d["input"])
    if not q:
        return None, "no_question"
    return {
        "question": q.group(1),
        "a1": out[: tc.end()],  # the model's own bytes, think + tool call
        "window": window,
        "resp": resp,
        "final": final,
        "answer": ans[0].strip(),
        "step": int(d["step"]),
        "score": float(d["score"]),
        "acc": float(d.get("acc", -1)),
        "evidence_iou": float(d.get("evidence_iou", -1)),
    }, ""


def render_row(t: dict, qrow: dict, frames_dir: Path) -> tuple[dict | None, str]:
    """Re-execute the logged crop through the RL tool's code path; the rebuilt
    response text must equal the logged one byte-for-byte or the trace drops."""
    duration = float(qrow["duration"])
    start, end, note = _normalize_window(t["window"][0], t["window"][1], duration, MIN_CROP_SECONDS)
    if start is None:
        return None, "tool_error_window"
    video_path = Path(qrow["video_path"])
    if not video_path.exists():
        return None, "missing_video"
    try:
        frames, timestamps = sample_frames(str(video_path), start, end, CROP_NUM_FRAMES,
                                           CROP_MAX_PIXELS, CROP_MIN_PIXELS)
    except Exception:
        return None, "decode_error"
    if crop_response_text(start, end, timestamps, note) != t["resp"]:
        return None, "response_text_mismatch"

    paths = []
    for i, fr in enumerate(frames):
        p = frames_dir / f"{video_path.stem}_{start:.1f}_{end:.1f}_{i}.jpg"
        if not p.exists():
            fr.save(p, quality=90)
        paths.append(str(p.resolve()))
    seg = qrow["video_segment"]
    messages = [
        {"role": "system", "content": build_system_prompt("tool_optional")},
        {"role": "user", "content": build_user_prompt(t["question"], duration)},
        {"role": "assistant", "content": strip_mm_tags(t["a1"])},
        {"role": "tool", "content": "<image>" * len(paths) + t["resp"]},
        {"role": "assistant", "content": strip_mm_tags(t["final"])},
    ]
    return {
        "messages": messages,
        "images": [{"image": p, "max_pixels": CROP_MAX_PIXELS, "min_pixels": CROP_MIN_PIXELS} for p in paths],
        "videos": [{"video": str(video_path.resolve()), "nframes": GLOBAL_NUM_FRAMES,
                    "max_pixels": GLOBAL_MAX_PIXELS, "min_pixels": GLOBAL_MIN_PIXELS}],
        "tools": [build_crop_video_schema().model_dump(exclude_unset=True, exclude_none=True)],
        "extra_info": {"question": t["question"], "video_id": video_path.name,
                       "source": "grpo_rollout", "tool_window": [start, end],
                       "duration": duration, "gt": qrow["gt_text"],
                       "video_segment": None if seg is None else [float(seg[0]), float(seg[1])],
                       "step": t["step"], "score": t["score"], "acc": t["acc"],
                       "evidence_iou": t["evidence_iou"]},
    }, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Round 2 default (2026-09-01). During a run verl writes results/<run>/rollouts/;
    # round 1's were renamed to rollouts_grpo267/ afterwards so a rerun under the
    # same EXP_NAME could not overwrite them file by file (DATA.md §0.5). Point
    # this at whichever name the finished run left behind.
    ap.add_argument("--rollouts", type=Path, default=Path("results/grpo-v2/rollouts"))
    ap.add_argument("--rl-train", type=Path, default=Path("data/processed/rl_train.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--min-score", type=float, default=1.5)
    ap.add_argument("--traces-per-q", type=int, default=3)
    ap.add_argument("--review-score", type=float, default=1.8, help="hand-read sample threshold")
    ap.add_argument("--review-n", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    # Every output is prefixed so a v2 build cannot silently overwrite the round-1
    # RFT set (whose parquets are the only record of what results/rft trained on).
    ap.add_argument("--prefix", default="rft_v2", help="output basename prefix (round 1 used 'rft')")
    ap.add_argument("--plan-only", action="store_true", help="selection + <prefix>_selection.json, no rendering")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- rl_train: question -> video/duration/gt (unique, README traceback) --
    rl = pd.read_parquet(args.rl_train)
    by_q: dict[str, dict] = {}
    for ei in rl["extra_info"]:
        ei = dict(ei)
        assert ei["question"] not in by_q, f"duplicate question in rl_train: {ei['question'][:60]}"
        by_q[ei["question"]] = ei

    # ---- collect + filter ----------------------------------------------------
    drop = Counter()
    traces: dict[str, list[dict]] = {}
    files = sorted(args.rollouts.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no <step>.jsonl under {args.rollouts}")
    n_rows = 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                n_rows += 1
                if float(d["score"]) <= args.min_score:
                    drop["low_score"] += 1
                    continue
                t, why = parse_rollout(d)
                if t is None:
                    drop[why] += 1
                    continue
                qrow = by_q.get(t["question"])
                if qrow is None:
                    drop["question_not_in_rl_train"] += 1
                    continue
                # The prompt the model saw must embed the duration we will
                # rebuild with; anything else means a stale rl_train.parquet.
                if f"The video is {float(qrow['duration']):.1f} seconds long." not in d["input"]:
                    drop["duration_mismatch"] += 1
                    continue
                if d.get("gts", "").strip() != str(qrow["gt_text"]).strip():
                    drop["gt_mismatch"] += 1
                    continue
                traces.setdefault(t["question"], []).append(t)

    # ---- dedupe + pick: late steps first, then answer diversity --------------
    picked_by_q: dict[str, list[dict]] = {}
    for q, ts in traces.items():
        seen, uniq = set(), []
        for t in sorted(ts, key=lambda t: (-t["step"], -t["score"])):
            key = (t["a1"], t["final"])
            if key in seen:
                drop["exact_duplicate"] += 1
                continue
            seen.add(key)
            uniq.append(t)
        picked_by_q[q] = [uniq[i] for i in pick_traces(uniq, args.traces_per_q)]
    picked = [t for ts in picked_by_q.values() for t in ts]

    steps = sorted(t["step"] for t in picked)
    sel = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rollout_rows": n_rows,
        "kept_traces": sum(len(v) for v in traces.values()),
        "questions": len(traces),
        "picked": len(picked),
        "picked_step_quartiles": [steps[0], steps[len(steps) // 4], steps[len(steps) // 2],
                                  steps[3 * len(steps) // 4], steps[-1]] if steps else [],
        "dropped": dict(drop),
        "per_question": {q: [{"step": t["step"], "score": t["score"], "answer": t["answer"]}
                             for t in ts] for q, ts in picked_by_q.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.prefix}_selection.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False))
    print(f"rollouts: {n_rows} rows -> {sel['kept_traces']} kept over {sel['questions']} questions "
          f"-> picked {len(picked)} (<= {args.traces_per_q}/question)")
    print(f"picked step quartiles: {sel['picked_step_quartiles']} | dropped: {dict(drop)}")

    # ---- hand-read sample (README: read score > review-score before training) --
    pool = [t for t in picked if t["score"] > args.review_score]
    take = rng.choice(len(pool), size=min(args.review_n, len(pool)), replace=False) if pool else []
    lines = [f"# RFT hand-read sample: {len(take)} of {len(pool)} picked traces with score > {args.review_score}",
             "", "Reject the extraction (or tighten --min-score) if answers look "
             "judge-flattered, thinks reference frames that cannot exist, or windows game IoU.", ""]
    for j, i in enumerate(sorted(take)):
        t = pool[i]
        g = by_q[t["question"]]
        lines += [f"## {j + 1}. step {t['step']} · score {t['score']:.3f} · acc {t['acc']:.0f} · "
                  f"iou {t['evidence_iou']:.3f}",
                  f"**Q:** {t['question']}", f"**GT:** {g['gt_text']}  ·  **segment:** {g['video_segment']}",
                  f"**model answer:** {t['answer']}", "", "```", t["a1"], "```", "",
                  f"tool → {t['resp'][:160]}…", "", "```", t["final"], "```", ""]
    (args.out / f"{args.prefix}_review.md").write_text("\n".join(lines))
    print(f"-> {args.out}/{args.prefix}_selection.json, {args.prefix}_review.md")
    if args.plan_only:
        return

    # ---- render --------------------------------------------------------------
    frames_dir = args.out / f"frames_{args.prefix}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, t in enumerate(picked, 1):
        row, why = render_row(t, by_q[t["question"]], frames_dir)
        if row is None:
            drop[why] += 1
        else:
            rows.append(row)
        if n % 200 == 0:
            print(f"  rendered {n}/{len(picked)} ({len(rows)} ok)")

    # Same guarantee as render_traces: every placeholder has an asset behind it,
    # positionally -- an off-by-one surfaces mid-epoch in a dataloader worker.
    for r in rows:
        n_img = sum(m["content"].count("<image>") for m in r["messages"])
        n_vid = sum(m["content"].count("<video>") for m in r["messages"])
        assert (n_img, n_vid) == (len(r["images"]), len(r["videos"])), (
            f"placeholder/asset mismatch in {r['extra_info']['video_id']}: "
            f"{n_img} <image> vs {len(r['images'])} images, "
            f"{n_vid} <video> vs {len(r['videos'])} videos")

    vids = sorted({r["extra_info"]["video_id"] for r in rows})
    val_vids = set(rng.choice(vids, size=max(1, int(len(vids) * args.val_frac)), replace=False))
    train = [r for r in rows if r["extra_info"]["video_id"] not in val_vids]
    val = [r for r in rows if r["extra_info"]["video_id"] in val_vids]
    pd.DataFrame(train).to_parquet(args.out / f"{args.prefix}_train.parquet")
    pd.DataFrame(val).to_parquet(args.out / f"{args.prefix}_val.parquet")
    sel["rendered"] = len(rows)
    sel["dropped"] = dict(drop)
    (args.out / f"{args.prefix}_selection.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False))
    print(f"rendered {len(rows)}/{len(picked)} (train {len(train)} / val {len(val)}), dropped {dict(drop)}")
    print(f"-> {args.out}/{args.prefix}_train.parquet, {args.prefix}_val.parquet, frames_{args.prefix}/ "
          f"({len(list(frames_dir.iterdir()))} jpgs)")


if __name__ == "__main__":
    main()
