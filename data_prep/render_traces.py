#!/usr/bin/env python
"""Build the QA SFT set from LongVT selftrace (+ optional geminicot mix).

Derivation (session 2026-08-25; supersedes the pure-TVG SFT for the QA pivot):

  selftrace 15,354 traces = LongVT's stage-3 RFT data: their model's successful
  RL rollouts (answer judged correct AND crop-vs-evidence IoU >= 0.3) over the
  selfqa questions. Only 1,290 unique (video, question) pairs; median 7
  duplicate solutions each.

  Join selftrace <-> selfqa on exact question text, then allocate by answer
  verifiability (normalized GT word count <= --max-gt-words):
    - joined & unverifiable        -> SFT (RL's matcher can't score them)
    - joined & verifiable          -> sample (weight 1/n_traces, favoring the
      questions their model rarely solved) until --sft-questions; rest -> RL
    - selfqa-only & verifiable     -> RL
    - selfqa-only & unverifiable   -> dropped (no traces, no verifiable reward)
    - selftrace-only (videos absent from selfqa_1.zip) -> dropped

  Every surface detail is re-rendered to match our RL rollout byte-for-byte
  (same rationale as the TVG renderer, render_tvg_traces.py): prompts rebuilt from agentic_tvg.prompts
  (QA mode), tool_call canonicalized (no video_path), tool-response frames
  re-decoded from the local mp4 via agentic_tvg.video_frames (their jpgs live
  in archives we do not download), val split by video id.

Outputs: allocation.json (always; consumed by extract_rl.py),
sft_train.parquet / sft_val.parquet + frames/*.jpg (unless --plan-only).

Usage:
    python data_prep/render_traces.py --plan-only     # selection stats only
    python data_prep/render_traces.py                 # full render
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_tvg.constants import (  # noqa: E402
    CROP_MAX_PIXELS,
    CROP_MIN_PIXELS,
    CROP_NUM_FRAMES,
    GLOBAL_MAX_PIXELS,
    GLOBAL_MIN_PIXELS,
    GLOBAL_NUM_FRAMES,
)
from agentic_tvg.crop_video_tool import build_crop_video_schema, crop_response_text  # noqa: E402
from agentic_tvg.prompts import build_system_prompt_qa, build_user_prompt_qa  # noqa: E402
from agentic_tvg.video_frames import get_video_duration, sample_frames  # noqa: E402

VID_RE = re.compile(r"([A-Za-z0-9_\-]+\.mp4)")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def norm_answer(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(_ARTICLES.sub(" ", s).split())


def msg_text(msg) -> str:
    c = msg["content"]
    if isinstance(c, str):
        return c
    return "".join(seg.get("text") or "" for seg in c if isinstance(seg, dict))


def parse_trace(msgs, source: str) -> dict | None:
    """(question, vid, think1, window, final) from one 5-message trace."""
    if len(msgs) != 5:
        return None
    u = msg_text(msgs[1])
    m = VID_RE.search(u)
    if not m:
        return None
    vid = m.group(1)
    if source == "selftrace":
        question = u.split("Think first")[0].replace("<video>", "").strip()
    else:  # geminicot: bare question + "The video path for this video is X.mp4"
        question = re.split(r"The [Vv]ideo path", u)[0].replace("<video>", "").strip()
    if not question:
        return None
    a1 = msg_text(msgs[2])
    tc = TOOL_CALL_RE.search(a1)
    if not tc:
        return None
    try:
        args = json.loads(tc.group(1))["arguments"]
        start, end = float(args["start_time"]), float(args["end_time"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    think1 = a1[: tc.start()].rstrip()
    if not (think1.startswith("<think>") and think1.endswith("</think>")):
        return None
    final = msg_text(msgs[4]).strip()
    ans = ANSWER_RE.findall(final)
    if len(ans) != 1 or not final.endswith("</answer>") or not final.startswith("<think>"):
        return None
    return {"question": question, "vid": vid, "think1": think1, "window": (start, end),
            "final": final, "answer": ans[0].strip()}


def canonical_tool_call(start: float, end: float) -> str:
    """Byte-identical to the Qwen3 chat template's own tool-call rendering."""
    payload = json.dumps({"name": "crop_video", "arguments": {"start_time": start, "end_time": end}})
    return f"<tool_call>\n{payload}\n</tool_call>"


def pick_traces(traces: list[dict], k: int) -> list[int]:
    """Indices of up to k traces, preferring distinct answers (diversity)."""
    by_ans: dict[str, list[int]] = {}
    for i, t in enumerate(traces):
        by_ans.setdefault(norm_answer(t["answer"]), []).append(i)
    picked, rounds = [], 0
    while len(picked) < k and rounds < k:
        for idxs in by_ans.values():
            if rounds < len(idxs) and len(picked) < k:
                picked.append(idxs[rounds])
        rounds += 1
    return sorted(picked)


def render_row(question, vid, think1, window, final, video_path, duration,
               frames_dir: Path, source: str, extra: dict) -> dict | None:
    start = max(0.0, min(window[0], duration - 0.5))
    end = min(window[1], duration)
    if end - start < 0.5:
        return None
    frames, timestamps = sample_frames(str(video_path), start, end, CROP_NUM_FRAMES,
                                       CROP_MAX_PIXELS, CROP_MIN_PIXELS)
    stem = Path(vid).stem
    paths = []
    for i, fr in enumerate(frames):
        p = frames_dir / f"{stem}_{start:.1f}_{end:.1f}_{i}.jpg"
        if not p.exists():
            fr.save(p, quality=90)
        paths.append(str(p.resolve()))
    messages = [
        {"role": "system", "content": build_system_prompt_qa("tool_optional")},
        {"role": "user", "content": build_user_prompt_qa(question, duration)},
        {"role": "assistant", "content": think1 + "\n" + canonical_tool_call(start, end)},
        {"role": "tool", "content": "<image>" * len(paths)
         + crop_response_text(start, end, [round(t, 2) for t in timestamps])},
        {"role": "assistant", "content": final},
    ]
    return {
        "messages": messages,
        "images": [{"image": p, "max_pixels": CROP_MAX_PIXELS, "min_pixels": CROP_MIN_PIXELS} for p in paths],
        "videos": [{"video": str(video_path.resolve()), "nframes": GLOBAL_NUM_FRAMES,
                    "max_pixels": GLOBAL_MAX_PIXELS, "min_pixels": GLOBAL_MIN_PIXELS}],
        "tools": [build_crop_video_schema().model_dump(exclude_unset=True, exclude_none=True)],
        "extra_info": {"question": question, "video_id": vid, "source": source,
                       "tool_window": [start, end], "duration": duration, **extra},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftrace", type=Path, default=Path("data/annotations/longvt_rft_selftrace_15k3.parquet"))
    ap.add_argument("--selfqa", type=Path, default=Path("data/annotations/longvt_rl_selfqa_1k6.parquet"))
    ap.add_argument("--geminicot", type=Path, default=Path("data/annotations/longvt_sft_geminicot_4k8.parquet"))
    ap.add_argument("--selfqa-video-root", type=Path, default=Path("data/videos/selfqa"))
    ap.add_argument("--geminicot-video-root", type=Path, default=Path("data/videos/geminicot"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--sft-questions", type=int, default=600, help="selftrace questions allocated to SFT")
    ap.add_argument("--traces-per-q", type=int, default=3)
    ap.add_argument("--geminicot-n", type=int, default=600, help="geminicot traces mixed in (0 = pure selftrace)")
    ap.add_argument("--max-gt-words", type=int, default=6, help="normalized GT length cap for 'verifiable'")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="selection + allocation.json, no rendering")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- selfqa: question -> (gt, video, segment) --------------------------
    sq = pd.read_parquet(args.selfqa)
    qa: dict[str, dict] = {}
    for _, r in sq.iterrows():
        ei = dict(r["extra_info"])
        seg = ei.get("video_segment")
        vids = r["videos"]
        vid = VID_RE.search(str(list(vids) if not isinstance(vids, str) else vids))
        qa[ei["question"].strip()] = {
            "gt": dict(r["reward_model"])["ground_truth"].strip(),
            "vid": vid.group(1) if vid else None,
            "segment": [float(seg[0]), float(seg[1])] if seg is not None else None,
        }

    # ---- selftrace: question -> traces ------------------------------------
    st = pd.read_parquet(args.selftrace)
    traces: dict[str, list[dict]] = {}
    bad_parse = 0
    for _, r in st.iterrows():
        t = parse_trace(r["messages"], "selftrace")
        if t is None:
            bad_parse += 1
            continue
        traces.setdefault(t["question"], []).append(t)

    joined = sorted(set(traces) & set(qa))
    selftrace_only = sorted(set(traces) - set(qa))
    selfqa_only = sorted(set(qa) - set(traces))
    verif = {q for q in qa if len(norm_answer(qa[q]["gt"]).split()) <= args.max_gt_words}

    # ---- allocation --------------------------------------------------------
    forced_sft = [q for q in joined if q not in verif]
    contested = [q for q in joined if q in verif]
    need = max(0, args.sft_questions - len(forced_sft))
    w = np.array([1.0 / len(traces[q]) for q in contested])
    sampled = list(rng.choice(contested, size=min(need, len(contested)),
                              replace=False, p=w / w.sum())) if need else []
    sft_qs = forced_sft + sampled
    rl_qs = sorted((set(contested) - set(sampled)) | (set(selfqa_only) & verif))
    dropped_unverif = sorted(set(selfqa_only) - verif)

    plan = {
        "config": vars(args) | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        "sft_selftrace": [
            {"question": q, "vid": qa[q]["vid"], "gt": qa[q]["gt"], "segment": qa[q]["segment"],
             "n_traces": len(traces[q]), "picked": pick_traces(traces[q], args.traces_per_q)}
            for q in sft_qs
        ],
        "rl": [{"question": q, "vid": qa[q]["vid"], "gt": qa[q]["gt"], "segment": qa[q]["segment"]}
               for q in rl_qs],
        "dropped": {"selftrace_only_questions": len(selftrace_only),
                    "selfqa_unverifiable_no_traces": len(dropped_unverif),
                    "selftrace_bad_parse": bad_parse},
    }

    # ---- geminicot mix -----------------------------------------------------
    gem_items = []
    if args.geminicot_n > 0:
        gm = pd.read_parquet(args.geminicot)
        pool = []
        for i, r in gm.iterrows():
            t = parse_trace(r["messages"], "geminicot")
            if t is not None:
                pool.append(t)
        take = rng.choice(len(pool), size=min(args.geminicot_n, len(pool)), replace=False)
        gem_items = [pool[i] for i in sorted(take)]
        plan["sft_geminicot"] = [{"question": t["question"], "vid": t["vid"]} for t in gem_items]
        plan["dropped"]["geminicot_bad_parse"] = len(gm) - len(pool)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "allocation.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))

    n_sft_traces = sum(len(x["picked"]) for x in plan["sft_selftrace"])
    print(f"selftrace: joined {len(joined)} | selftrace-only {len(selftrace_only)} (dropped) | bad-parse {bad_parse}")
    print(f"SFT: {len(forced_sft)} forced(unverifiable) + {len(sampled)} sampled = {len(sft_qs)} questions "
          f"-> {n_sft_traces} traces | + geminicot {len(gem_items)}")
    print(f"RL : {len(rl_qs)} verifiable questions | dropped unverifiable-no-trace {len(dropped_unverif)}")
    print(f"-> {args.out}/allocation.json")
    if args.plan_only:
        return

    # ---- render ------------------------------------------------------------
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows, drop = [], Counter()
    durations: dict[str, float] = {}

    def duration_of(path: Path) -> float:
        key = str(path)
        if key not in durations:
            try:
                durations[key] = get_video_duration(key)
            except Exception:
                durations[key] = -1.0
        return durations[key]

    jobs = []
    for item in plan["sft_selftrace"]:
        q = item["question"]
        for i in item["picked"]:
            t = traces[q][i]
            jobs.append((t, args.selfqa_video_root / t["vid"], "selftrace",
                         {"gt": item["gt"], "video_segment": item["segment"]}))
    for t in gem_items:
        jobs.append((t, args.geminicot_video_root / t["vid"], "geminicot",
                     {"gt": None, "video_segment": None}))

    for t, vpath, source, extra in jobs:
        if not vpath.exists():
            drop["missing_video"] += 1
            continue
        dur = duration_of(vpath)
        if dur <= 0:
            drop["bad_duration"] += 1
            continue
        try:
            row = render_row(t["question"], t["vid"], t["think1"], t["window"],
                             t["final"], vpath, dur, frames_dir, source, extra)
        except Exception:
            drop["decode_error"] += 1
            continue
        if row is None:
            drop["bad_window"] += 1
            continue
        rows.append(row)

    vids = sorted({r["extra_info"]["video_id"] for r in rows})
    val_vids = set(rng.choice(vids, size=max(1, int(len(vids) * args.val_frac)), replace=False))
    train = [r for r in rows if r["extra_info"]["video_id"] not in val_vids]
    val = [r for r in rows if r["extra_info"]["video_id"] in val_vids]
    pd.DataFrame(train).to_parquet(args.out / "sft_train.parquet")
    pd.DataFrame(val).to_parquet(args.out / "sft_val.parquet")
    print(f"rendered {len(rows)}/{len(jobs)} (train {len(train)} / val {len(val)}), dropped {dict(drop)}")
    print(f"-> {args.out}/sft_train.parquet, sft_val.parquet, frames/ ({len(list(frames_dir.iterdir()))} jpgs)")


if __name__ == "__main__":
    main()
